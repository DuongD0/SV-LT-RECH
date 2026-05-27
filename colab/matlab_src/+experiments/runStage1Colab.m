function summary = runStage1Colab(configPath, opts)
% runStage1Colab  Colab-adapted Stage 1 simulation study runner.
%
%   summary = experiments.runStage1Colab(configPath)
%   summary = experiments.runStage1Colab(configPath, opts)
%
%   Differences from experiments.runStage1 (local):
%     * Uses utils.loadConfigColab (no Python interop required).
%     * RESUMABLE across Colab sessions: if recovery.csv already
%       contains rows for replicates 1..R0, those replicates are
%       skipped and the run continues from R0+1. SMC is seeded
%       deterministically off baseSeed so resumed results are
%       bit-identical to a single-session run.
%     * Saves per-(replicate, seed) posterior .mat for Phase 4 warm-starts.
%     * Saves per-replicate simulated y + ground-truth h .mat so Phase 4
%       can rerun any chain without redoing the DGP.
%     * Writes results/gate_summary.json with §4.5 verdicts after the
%       last replicate.
%
%   Inputs
%   ------
%     configPath : path to stage1_colab.yaml (or stage1_colab_quick.yaml).
%     opts       : struct, optional
%                    .outDir         output directory (default 'results')
%                    .writeOutputs   true (default)
%                    .resume         true (default) -- skip replicates already
%                                    present in recovery.csv
%
%   See PHASE_4_HANDOFF.md for the spec of what comes out of this run.

    arguments
        configPath (1,1) string
        opts       struct = struct()
    end

    defaults.outDir       = fullfile('results');
    defaults.writeOutputs = true;
    defaults.resume       = true;
    opts = mergeStruct(defaults, opts);

    fprintf('[stage1-colab] config: %s\n', configPath);
    fprintf('[stage1-colab] outDir: %s\n', opts.outDir);
    fprintf('[stage1-colab] resume: %s\n', mat2str(opts.resume));

    %% Resolve config paths relative to the +experiments package parent.
    %% Accept either .json or .yaml-spelled paths; we always read JSON
    %% so the .yaml input is silently rewritten to .json.
    repoRoot   = fileparts(fileparts(mfilename('fullpath')));
    smcDefault = fullfile(repoRoot, 'config', 'smc_defaults.json');

    configPath = char(configPath);
    if endsWith(configPath, '.yaml')
        configPath = [configPath(1:end-5), '.json'];
    end
    if ~isfile(configPath)
        configPath = fullfile(repoRoot, configPath);
    end
    cfg = utils.loadConfigColab(smcDefault, configPath);

    %% Build model
    thetaTrue = cell2mat(cfg.dgp.theta_true);
    K         = cfg.dgp.nCovariates;
    leverage  = cfg.dgp.leverage;
    mdl       = models.SVLTRECH('nCovariates', K, 'leverage', leverage);
    pnames    = mdl.paramNames();

    expectedLen = 12 + K;
    assert(numel(thetaTrue) == expectedLen, ...
        'runStage1Colab:thetaLen', ...
        'theta_true length %d != 12 + nCovariates(%d)', numel(thetaTrue), K);

    %% Output dirs
    figDir   = fullfile(opts.outDir, 'figures');
    postDir  = fullfile(opts.outDir, 'posteriors');
    simDir   = fullfile(opts.outDir, 'simulations');
    if opts.writeOutputs
        for d = {opts.outDir, figDir, postDir, simDir}
            if ~isfolder(d{1}); mkdir(d{1}); end
        end
    end

    R    = cfg.simulation.nReplicates;
    S    = cfg.simulation.nSeeds;
    base = cfg.simulation.baseSeed;

    %% Colab-specific persistence flags
    savePosteriors  = isfield(cfg, 'colab') && isfield(cfg.colab, 'savePosteriors') ...
                      && cfg.colab.savePosteriors;
    saveSimulations = isfield(cfg, 'colab') && isfield(cfg.colab, 'saveSimulations') ...
                      && cfg.colab.saveSimulations;

    %% Resume detection: scan recovery.csv for completed replicates.
    recoveryPath = fullfile(opts.outDir, 'recovery.csv');
    logZPath     = fullfile(opts.outDir, 'mcse_logml.csv');
    signPath     = fullfile(opts.outDir, 'sign_switch.csv');

    rowsRecovery   = cell(0, 8);
    rowsLogZ       = zeros(R, 2);
    rowsSignSwitch = zeros(R * S, 3);
    rowsIdx        = 0;

    rhatPerReplicate     = nan(R, expectedLen);
    postMeanPerReplicate = nan(R, expectedLen);
    postStdPerReplicate  = nan(R, expectedLen);

    startReplicate = 1;
    if opts.resume && opts.writeOutputs && isfile(recoveryPath)
        prior = readtable(recoveryPath);
        if ~isempty(prior)
            startReplicate = max(prior.replicate) + 1;
            fprintf('[stage1-colab] RESUMING from replicate %d (found %d rows in recovery.csv)\n', ...
                    startReplicate, height(prior));

            % Replay prior rows so the aggregate metrics include them.
            for i = 1:height(prior)
                rowsRecovery(end+1, :) = ...   %#ok<AGROW>
                    {prior.replicate(i), prior.seed(i), char(prior.param{i}), ...
                     prior.true(i), prior.postMean(i), prior.postStd(i), ...
                     prior.withinCI95(i), prior.rHat(i)};
                rr = prior.replicate(i);
                kIdx = find(strcmp(pnames, char(prior.param{i})), 1);
                if ~isempty(kIdx)
                    rhatPerReplicate(rr, kIdx)     = prior.rHat(i);
                    postMeanPerReplicate(rr, kIdx) = prior.postMean(i);
                    postStdPerReplicate(rr, kIdx)  = prior.postStd(i);
                end
            end
        end
        if isfile(logZPath)
            priorLogZ = readtable(logZPath);
            for i = 1:height(priorLogZ)
                rowsLogZ(priorLogZ.replicate(i), :) = ...
                    [priorLogZ.logZ_mean(i), priorLogZ.logZ_mcse(i)];
            end
        end
        if isfile(signPath)
            priorSign = readtable(signPath);
            for i = 1:height(priorSign)
                rowsIdx = rowsIdx + 1;
                rowsSignSwitch(rowsIdx, :) = ...
                    [priorSign.replicate(i), priorSign.seed(i), priorSign.fracFlipped(i)];
            end
        end
    end

    smcOpts = struct( ...
        'N',             cfg.smc.N, ...
        'M',             cfg.smc.M, ...
        'nSweeps',       cfg.smc.nSweeps, ...
        'proposalScale', cfg.smc.proposalScale, ...
        'essThreshold',  cfg.smc.essThreshold, ...
        'targetEss',     cfg.smc.targetEss, ...
        'verbose',       cfg.smc.verbose, ...
        'pfOpts',        struct('clipLogWeight', cfg.particleFilter.clipLogWeight));

    beta1Idx = 7;

    for r = startReplicate:R
        ticReplicate = tic;
        fprintf('[stage1-colab] replicate %d/%d  start  (wall %s)\n', ...
                r, R, datestr(now, 'yyyy-mm-dd HH:MM:SS'));

        rng(base + 1000 * r, 'threefry');
        if K > 0
            Z = randn(cfg.dgp.T, K);
            mdl.setCovariates(Z);
        else
            Z = [];
        end
        ticSim = tic;
        [y, hTrue] = mdl.simulate(thetaTrue, cfg.dgp.T);
        fprintf('[stage1-colab] rep %d  DGP simulate done in %.1f s (T=%d)\n', ...
                r, toc(ticSim), cfg.dgp.T);

        if saveSimulations && opts.writeOutputs
            save(fullfile(simDir, sprintf('replicate%d.mat', r)), ...
                 'y', 'hTrue', 'Z', 'thetaTrue', '-v7');
        end

        thetaChains = zeros(cfg.smc.N, expectedLen, S);
        logZChains  = zeros(S, 1);

        for c = 1:S
            rng(base + 1000 * r + c, 'threefry');
            ticChain = tic;
            fprintf('[stage1-colab] rep %d  chain %d/%d  start  (N=%d M=%d nSweeps=%d)\n', ...
                    r, c, S, cfg.smc.N, cfg.smc.M, cfg.smc.nSweeps);
            result = inference.smc.likelihoodAnneal(mdl, y, smcOpts);
            fprintf('[stage1-colab] rep %d  chain %d/%d  done in %.1f s  logZ=%.2f  K_anneal=%d\n', ...
                    r, c, S, toc(ticChain), result.logMarginalLik, numel(result.schedule));
            thetaChains(:, :, c) = result.theta;
            logZChains(c)        = result.logMarginalLik;

            [~, ssReport] = diagnostics.relabelSignSwitch(result.theta, beta1Idx);
            rowsIdx = rowsIdx + 1;
            rowsSignSwitch(rowsIdx, :) = [r, c, ssReport.fracFlipped];

            if savePosteriors && opts.writeOutputs
                outPost = struct( ...
                    'theta',          result.theta, ...
                    'logLik',         result.logLik, ...
                    'logMarginalLik', result.logMarginalLik, ...
                    'schedule',       result.schedule, ...
                    'essTrace',       result.essTrace, ...
                    'acceptTrace',    result.acceptTrace, ...
                    'paramNames',     {pnames}); %#ok<NASGU>
                save(fullfile(postDir, sprintf('replicate%d_seed%d.mat', r, c)), ...
                     '-struct', 'outPost', '-v7');
            end

            if opts.writeOutputs && r == startReplicate && c == 1
                fig = diagnostics.posteriorTrace(result, pnames, ...
                    struct('thetaTrue', thetaTrue, ...
                           'title', sprintf('Stage1 rep%d seed%d', r, c)));
                exportgraphics(fig, fullfile(figDir, ...
                    sprintf('F11_replicate%d_posterior.png', r)));
                close(fig);
            end
        end

        rhat = diagnostics.rHat(thetaChains);
        rhatPerReplicate(r, :) = rhat;

        thetaPooled = reshape(permute(thetaChains, [1, 3, 2]), [], expectedLen);
        postMean    = mean(thetaPooled, 1);
        postStd     = std(thetaPooled, 0, 1);
        postMeanPerReplicate(r, :) = postMean;
        postStdPerReplicate(r, :)  = postStd;

        for k = 1:expectedLen
            within95 = abs(postMean(k) - thetaTrue(k)) <= 2 * postStd(k);
            rowsRecovery(end+1, :) = ...   %#ok<AGROW>
                {r, NaN, pnames{k}, thetaTrue(k), postMean(k), postStd(k), within95, rhat(k)};
        end

        [mcse, meanLogZ] = diagnostics.mcseLogML(logZChains);
        rowsLogZ(r, :) = [meanLogZ, mcse];

        fprintf('[stage1-colab] replicate %d/%d  done in %.1f s  Rhat_max=%.3f  MCSE=%.3f\n', ...
                r, R, toc(ticReplicate), max(rhat, [], 'omitnan'), mcse);

        if opts.writeOutputs
            recoveryTblPartial = cell2table(rowsRecovery, ...
                'VariableNames', {'replicate','seed','param','true','postMean','postStd','withinCI95','rHat'});
            writetable(recoveryTblPartial, recoveryPath);

            logZTblPartial = array2table(rowsLogZ(1:r, :), 'VariableNames', {'logZ_mean','logZ_mcse'});
            logZTblPartial.replicate = (1:r)';
            writetable(logZTblPartial, logZPath);

            signTblPartial = array2table(rowsSignSwitch(1:rowsIdx, :), ...
                'VariableNames', {'replicate','seed','fracFlipped'});
            writetable(signTblPartial, signPath);
        end
    end

    %% Aggregate + §4.5 gate verdict
    coverage95 = zeros(1, expectedLen);
    for k = 1:expectedLen
        coverage95(k) = mean(abs(postMeanPerReplicate(:, k) - thetaTrue(k)) ...
                             <= 2 * postStdPerReplicate(:, k), 'omitnan');
    end

    summaryStruct.coverage95     = coverage95;
    summaryStruct.maxRhat        = max(rhatPerReplicate(:), [], 'omitnan');
    summaryStruct.maxFracFlipped = max(rowsSignSwitch(1:rowsIdx, 3));
    summaryStruct.mcseLogZ_max   = max(rowsLogZ(:, 2), [], 'omitnan');
    summaryStruct.meanLogZ       = rowsLogZ(:, 1);

    if opts.writeOutputs
        writeGateSummary(opts.outDir, configPath, pnames, coverage95, ...
                         summaryStruct.maxRhat, summaryStruct.mcseLogZ_max, ...
                         summaryStruct.maxFracFlipped, K);
        fprintf('[stage1-colab] wrote %s/{recovery,mcse_logml,sign_switch}.csv + gate_summary.json\n', ...
                opts.outDir);
    end

    out.summary              = summaryStruct;
    out.pnames               = pnames;
    out.thetaTrue            = thetaTrue;
    out.rhatPerReplicate     = rhatPerReplicate;
    out.postMeanPerReplicate = postMeanPerReplicate;
    out.postStdPerReplicate  = postStdPerReplicate;
    out.fracFlipped          = rowsSignSwitch(1:rowsIdx, :);

    summary = out;
end


function writeGateSummary(outDir, configPath, pnames, coverage95, ...
                          maxRhat, mcseLogZ, maxFracFlipped, K)
% Render the §4.5 acceptance criteria as a JSON file. v_z gates are
% only meaningful when nCovariates > 0; otherwise null.

    coverageMin = min(coverage95);
    gates = struct();
    gates.coverage_95_min = makeGate(coverageMin, 0.90, 'ge');
    gates.rhat_max        = makeGate(maxRhat,     1.10, 'le');
    gates.mcse_logz_max   = makeGate(mcseLogZ,    0.50, 'le');
    gates.sign_flip_max   = makeGate(maxFracFlipped, 0.05, 'lt');

    if K > 0
        vzIdx = find(startsWith(string(pnames), 'v_z_'));
        gates.v_z_indices    = vzIdx(:)';
        gates.v_z_coverage95 = coverage95(vzIdx);
    end

    overall = all([gates.coverage_95_min.pass, ...
                   gates.rhat_max.pass, ...
                   gates.mcse_logz_max.pass, ...
                   gates.sign_flip_max.pass]);

    payload = struct( ...
        'stage1_complete_at',    datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
        'config_file',           char(configPath), ...
        'paramNames',            {pnames}, ...
        'coverage95_per_param',  coverage95, ...
        'gates',                 gates, ...
        'overall',               logicalToVerdict(overall), ...
        'next_step',             phaseFourGuidance(overall));

    fid = fopen(fullfile(outDir, 'gate_summary.json'), 'w');
    fwrite(fid, jsonencode(payload, 'PrettyPrint', true));
    fclose(fid);
end


function g = makeGate(value, target, op)
    g.value = value;
    switch op
        case 'le'; g.target = sprintf('<= %g', target); g.pass = value <= target;
        case 'lt'; g.target = sprintf('< %g',  target); g.pass = value <  target;
        case 'ge'; g.target = sprintf('>= %g', target); g.pass = value >= target;
        case 'gt'; g.target = sprintf('> %g',  target); g.pass = value >  target;
        otherwise; error('makeGate:badOp', 'unknown op %s', op);
    end
end


function v = logicalToVerdict(b)
    if b; v = 'PASS'; else; v = 'FAIL'; end
end


function g = phaseFourGuidance(allPass)
    if allPass
        g = 'All §4.5 gates passed. Proceed to Phase 4 (LSTM/GRU cells, RealRECH, GARCH baselines).';
    else
        g = ['One or more §4.5 gates failed. Inspect recovery.csv + ' ...
             'mcse_logml.csv before launching Phase 4. Likely fix points: ' ...
             '(a) switch leverage to ''ocsn'' via stage1_colab.yaml; ' ...
             '(b) increase N or M in smc block; ' ...
             '(c) build the auxiliary PF (+inference/+pf/auxiliary.m) for fatter-tailed regimes.'];
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
