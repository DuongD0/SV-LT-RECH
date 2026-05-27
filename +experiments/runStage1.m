function summary = runStage1(configPath, opts)
% runStage1  Body of the Stage 1 simulation study (PROPOSED_METHODOLOGY.md §11.1).
%
%   summary = experiments.runStage1(configPath)
%   summary = experiments.runStage1(configPath, opts)
%
%   Reads a stage1*.yaml config, runs nReplicates x nSeeds SMC fits on
%   simulated SVLTRECH data, and returns aggregate metrics. Optionally
%   writes per-replicate CSVs + a sample figure to results/stage1/.
%
%   Inputs
%   ------
%     configPath : path to a YAML config (stage1.yaml or stage1_smoke.yaml).
%     opts       : struct, optional
%                    .writeOutputs   true (default) to write CSVs / figures
%                    .outDir         output directory (default 'results/stage1')
%
%   Output struct (always returned)
%   -------------------------------
%     .summary.coverage95       1-by-K  fraction of replicates with
%                                       |postMean - true| <= 2*postStd
%     .summary.maxRhat          scalar  max over replicates and params
%     .summary.maxFracFlipped   scalar  max sign-switch fraction
%     .summary.mcseLogZ_max     scalar  max MCSE of log Z across replicates
%     .summary.meanLogZ         R-by-1  mean log Z per replicate
%     .pnames                   1-by-K  parameter names
%     .thetaTrue                1-by-K  true theta used in DGP
%     .rhatPerReplicate         R-by-K
%     .postMeanPerReplicate     R-by-K
%     .postStdPerReplicate      R-by-K
%     .fracFlipped              R*S-by-3 [replicate, seed, frac]

    arguments
        configPath (1,1) string
        opts       struct = struct()
    end

    defaults.writeOutputs = true;
    defaults.outDir       = fullfile('results', 'stage1');
    opts = mergeStruct(defaults, opts);

    fprintf('[stage1] config: %s\n', configPath);

    % Resolve config paths relative to the repo root (the parent of this
    % +experiments package directory). Lets the function run regardless
    % of the caller's cwd (e.g. when invoked from `runtests('tests')`).
    repoRoot   = fileparts(fileparts(mfilename('fullpath')));
    smcDefault = fullfile(repoRoot, 'config', 'smc_defaults.yaml');
    if ~isfile(char(configPath))
        configPath = fullfile(repoRoot, char(configPath));
    end
    cfg = utils.loadConfig(smcDefault, char(configPath));

    %% Build model
    thetaTrue = cell2mat(cfg.dgp.theta_true);
    K         = cfg.dgp.nCovariates;
    leverage  = cfg.dgp.leverage;
    mdl       = models.SVLTRECH('nCovariates', K, 'leverage', leverage);
    pnames    = mdl.paramNames();

    expectedLen = 12 + K;
    assert(numel(thetaTrue) == expectedLen, ...
        'runStage1:thetaLen', ...
        'theta_true length %d != 12 + nCovariates(%d)', numel(thetaTrue), K);

    figDir = fullfile(opts.outDir, 'figures');
    if opts.writeOutputs
        if ~isfolder(opts.outDir); mkdir(opts.outDir); end
        if ~isfolder(figDir);      mkdir(figDir);      end
    end

    R    = cfg.simulation.nReplicates;
    S    = cfg.simulation.nSeeds;
    base = cfg.simulation.baseSeed;

    rowsRecovery   = cell(0, 8);
    rowsLogZ       = zeros(R, 2);
    rowsSignSwitch = zeros(R * S, 3);
    rowsIdx        = 0;

    rhatPerReplicate     = zeros(R, expectedLen);
    postMeanPerReplicate = zeros(R, expectedLen);
    postStdPerReplicate  = zeros(R, expectedLen);

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

    %% Incremental persistence handles -- written at end of each replicate
    %% so a crash mid-run preserves completed work.
    if opts.writeOutputs
        recoveryPath = fullfile(opts.outDir, 'recovery.csv');
        logZPath     = fullfile(opts.outDir, 'mcse_logml.csv');
        signPath     = fullfile(opts.outDir, 'sign_switch.csv');
        if isfile(recoveryPath); delete(recoveryPath); end
        if isfile(logZPath);     delete(logZPath);     end
        if isfile(signPath);     delete(signPath);     end
    end

    for r = 1:R
        ticReplicate = tic;
        fprintf('[stage1] replicate %d/%d  (elapsed total = %s)\n', ...
                r, R, datestr(now, 'HH:MM:SS'));

        rng(base + 1000 * r, 'threefry');
        if K > 0
            Z = randn(cfg.dgp.T, K);
            mdl.setCovariates(Z);
        end
        [y, ~] = mdl.simulate(thetaTrue, cfg.dgp.T);

        thetaChains = zeros(cfg.smc.N, expectedLen, S);
        logZChains  = zeros(S, 1);

        for c = 1:S
            rng(base + 1000 * r + c, 'threefry');
            result = inference.smc.likelihoodAnneal(mdl, y, smcOpts);
            thetaChains(:, :, c) = result.theta;
            logZChains(c)        = result.logMarginalLik;

            [~, ssReport] = diagnostics.relabelSignSwitch(result.theta, beta1Idx);
            rowsIdx = rowsIdx + 1;
            rowsSignSwitch(rowsIdx, :) = [r, c, ssReport.fracFlipped];

            if opts.writeOutputs && r == 1 && c == 1
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
            rowsRecovery(end+1, :) = ...     %#ok<AGROW>
                {r, NaN, pnames{k}, thetaTrue(k), postMean(k), postStd(k), within95, rhat(k)};
        end

        [mcse, meanLogZ] = diagnostics.mcseLogML(logZChains);
        rowsLogZ(r, :) = [meanLogZ, mcse];

        fprintf('[stage1] replicate %d/%d done in %.1f s (Rhat max %.3f, MCSE %.3f)\n', ...
                r, R, toc(ticReplicate), max(rhat, [], 'omitnan'), mcse);

        %% Persist incrementally
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

    if opts.writeOutputs
        fprintf('[stage1] wrote %s/{recovery,mcse_logml,sign_switch}.csv\n', opts.outDir);
    end

    coverage95 = zeros(1, expectedLen);
    for k = 1:expectedLen
        coverage95(k) = mean(abs(postMeanPerReplicate(:, k) - thetaTrue(k)) ...
                             <= 2 * postStdPerReplicate(:, k));
    end

    summaryStruct.coverage95     = coverage95;
    summaryStruct.maxRhat        = max(rhatPerReplicate(:), [], 'omitnan');
    summaryStruct.maxFracFlipped = max(rowsSignSwitch(1:rowsIdx, 3));
    summaryStruct.mcseLogZ_max   = max(rowsLogZ(:, 2), [], 'omitnan');
    summaryStruct.meanLogZ       = rowsLogZ(:, 1);

    out.summary              = summaryStruct;
    out.pnames               = pnames;
    out.thetaTrue            = thetaTrue;
    out.rhatPerReplicate     = rhatPerReplicate;
    out.postMeanPerReplicate = postMeanPerReplicate;
    out.postStdPerReplicate  = postStdPerReplicate;
    out.fracFlipped          = rowsSignSwitch(1:rowsIdx, :);

    summary = out;
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
