function out = runStage4(y, Z, splitIdx, cfg, opts)
% runStage4  Stage-4 empirical pipeline: fit 6 models, OOS-forecast, score,
%            compare, and assemble a complete results bundle.
%
%   out = experiments.runStage4(y, Z, splitIdx, cfg)
%   out = experiments.runStage4(y, Z, splitIdx, cfg, opts)
%
%   Loops over experiments.stage4Models(): GARCH-t, GJR-t (Econometrics MLE +
%   garch.rollingForecast); SVLT, SVLTRECH-{SRN,LSTM,GRU} (likelihoodAnneal +
%   inference.forecast.rollingPredictive). Scores PPS/QS/MSE/MAE/R2LOG/QLIKE on
%   squared-return proxy; Diebold-Mariano each baseline vs the proposed model
%   (SVLTRECH-SRN) and the Model Confidence Set on per-step QLIKE. Persists a
%   self-describing bundle (design spec §4.1) consumed by the +report layer.
%
%   cfg fields: .model(.nCovariates,.leverage), .smc(.N,.M,.nSweeps,
%   .proposalScale,.essThreshold,.targetEss,.verbose), .forecast.J,
%   .eval(.alphaQS,.mcsB), .particleFilter.clipLogWeight, .baseSeed.
%   opts (optional): .writeOutputs (false), .outDir ('results/stage4'),
%                    .expName ('stage4').
%
%   Output `out`: .modelNames, .scores (struct array), .dmVsProposed,
%   .mcs, .forecasts, .testIdx, .bundle (the full persisted struct).

    arguments
        y        (:,1) double
        Z        (:,:) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
        cfg      (1,1) struct
        opts           struct = struct()
    end

    defaults.writeOutputs = false;
    defaults.outDir       = fullfile('results', 'stage4');
    defaults.expName      = 'stage4';
    opts = mergeStruct(defaults, opts);

    utils.reproducibility(cfg.baseSeed);

    %% Mean equation (ARMA auto-gate, PROPOSED_METHODOLOGY.md §3). Every
    %% variance model is fit on the mean-equation RESIDUALS so the
    %% conditional-variance comparison is not contaminated by serial
    %% correlation in the conditional mean. White-noise returns pass
    %% through unchanged (zero-mean). Override via cfg.mean.force.
    yRaw          = y;
    [y, meanInfo] = applyMeanEquation(yRaw, splitIdx, cfg);

    T       = numel(y);
    testIdx = (splitIdx + 1 : T)';
    yTest   = y(testIdx);
    rvProxy = max(yTest .^ 2, 1e-8);
    rvSqrt  = sqrt(rvProxy);
    alphaQS = cfg.eval.alphaQS;
    K       = size(Z, 2);

    smcOpts = struct( ...
        'N', cfg.smc.N, 'M', cfg.smc.M, 'nSweeps', cfg.smc.nSweeps, ...
        'proposalScale', cfg.smc.proposalScale, 'essThreshold', cfg.smc.essThreshold, ...
        'targetEss', cfg.smc.targetEss, 'verbose', cfg.smc.verbose, ...
        'pfOpts', struct('clipLogWeight', cfg.particleFilter.clipLogWeight));
    pfOpts = struct('clipLogWeight', cfg.particleFilter.clipLogWeight);
    J      = cfg.forecast.J;

    reg        = experiments.stage4Models();
    nModels    = numel(reg);
    modelNames = {reg.name};
    proposedIx = find([reg.isProposed], 1);

    scores  = repmat(emptyScore(), 1, nModels);
    qlAll   = zeros(numel(testIdx), nModels);
    models_ = repmat(emptyModelRecord(), 1, nModels);
    fc      = struct();

    for m = 1:nModels
        rec = fitOneModel(reg(m), y, Z, splitIdx, K, smcOpts, pfOpts, J, cfg);
        [scores(m), qlAll(:, m)] = scoreModel( ...
            rec.varForecast, rec.logPredDensity, rec.nu, yTest, rvProxy, rvSqrt, alphaQS);
        models_(m) = rec;
        fc.(matlab.lang.makeValidName(reg(m).name)) = rec.varForecast;
    end

    %% Diebold-Mariano: each non-proposed model vs the proposed model.
    others = setdiff(1:nModels, proposedIx);
    dmVsProposed = repmat(eval.dieboldMariano(qlAll(:,others(1)), qlAll(:,proposedIx), 1), ...
                          1, numel(others));
    for k = 1:numel(others)
        dmVsProposed(k) = eval.dieboldMariano(qlAll(:, others(k)), qlAll(:, proposedIx), 1);
    end
    dmBaselines = modelNames(others);

    %% Model Confidence Set on per-step QLIKE loss.
    mcs = eval.modelConfidenceSet(qlAll, struct('B', cfg.eval.mcsB));

    %% In-sample descriptives + diagnostics (T1 record) on the RAW returns.
    descr = describeSeries(yRaw(1:splitIdx));

    %% Assemble bundle (design spec §4.1).
    bundle.meta = struct('expName', opts.expName, ...
                         'date', char(datetime('now', 'Format', 'yyyy-MM-dd')), ...
                         'cfg', cfg, 'baseSeed', cfg.baseSeed, 'modelNames', {modelNames});
    bundle.data = struct('y', yRaw, 'resid', y, 'Z', Z, 'splitIdx', splitIdx, ...
                         'testIdx', testIdx, 'nCovariates', K, 'meanEquation', meanInfo);
    bundle.descriptives = descr;
    bundle.models       = models_;
    bundle.comparison   = struct('modelNames', {modelNames}, 'scores', scores, ...
                                 'qlAll', qlAll, 'dmVsProposed', dmVsProposed, ...
                                 'dmBaselines', {dmBaselines}, 'mcs', mcs, ...
                                 'proposedName', modelNames{proposedIx});

    out.modelNames   = modelNames;
    out.scores       = scores;
    out.dmVsProposed = dmVsProposed;
    out.mcs          = mcs;
    out.forecasts    = fc;
    out.testIdx      = testIdx;
    out.bundle       = bundle;

    if opts.writeOutputs
        writeStage4Outputs(bundle, opts.outDir);
    end
end


function rec = fitOneModel(spec, y, Z, splitIdx, K, smcOpts, pfOpts, J, cfg)
% Fit one registry entry; return a fully-populated model record.
    rec = emptyModelRecord();
    rec.name = spec.name;
    rec.type = spec.kind;

    if strcmp(spec.kind, 'garch')
        fit = garch.fitGarch(y(1:splitIdx), struct('type', spec.garchType, 'dist', 't'));
        rf  = garch.rollingForecast(fit, y, splitIdx);
        rec.nu             = fit.nu;
        rec.varForecast    = rf.sigma2Forecast;
        rec.logPredDensity = rf.logPredDensity;
        rec.logMarginalLik = NaN;                       % MLE, not SMC
        rec.sigma2InSample = fit.condVar;
        rec.stdResid       = standardizeResid(y(1:splitIdx), fit.condVar, fit.nu);
        rec.omegaPath      = [];
        [rec.paramNames, rec.posteriorMean, rec.posteriorStd] = garchParams(fit, spec.garchType);
        rec.theta          = [];
        return;
    end

    if strcmp(spec.kind, 'garchrech')
        % GARCH-RECH deep-learning baseline (MLE; reuses +cells via garch.*).
        grOpts = struct('cell', spec.cell, 'dist', 't', ...
                        'restarts', 3, 'maxEval', 4000, 'verbose', false);
        if isfield(cfg, 'garchRech'); grOpts = mergeStruct(grOpts, cfg.garchRech); end
        if K > 0; grOpts.Z = Z(1:splitIdx, :); else; grOpts.Z = []; end

        fit = garch.fitGarchRECH(y(1:splitIdx), grOpts);
        rf  = garch.rollingForecastRECH(fit, y, Z, splitIdx);
        rec.nu             = fit.nu;
        rec.varForecast    = rf.sigma2Forecast;
        rec.logPredDensity = rf.logPredDensity;
        rec.logMarginalLik = NaN;                       % MLE, not SMC
        rec.sigma2InSample = fit.condVar;
        rec.stdResid       = standardizeResid(y(1:splitIdx), fit.condVar, fit.nu);
        rec.omegaPath      = fit.omegaPath;
        [rec.paramNames, rec.posteriorMean, rec.posteriorStd] = garchRechParams(fit);
        rec.theta          = [];
        return;
    end

    % --- SV / RECH model (SMC) ---
    mdl = spec.ctor(K, cfg.model.leverage);
    if ismethod(mdl, 'setCovariates') && mdl.nCovariates > 0
        mdl.setCovariates(Z(1:splitIdx, :));
    end
    res      = inference.smc.likelihoodAnneal(mdl, y(1:splitIdx), smcOpts);
    thetaBar = mean(res.theta, 1);
    pn       = mdl.paramNames();
    nuIdx    = find(strcmp(pn, 'nu'), 1);
    if isempty(nuIdx); nu = Inf; else; nu = thetaBar(nuIdx); end   % Gaussian (SV/SVM) -> Inf

    % In-sample filtered states -> sigma2, std residuals, omega (RECH only).
    [~, hF] = inference.pf.bootstrap(mdl, y(1:splitIdx), thetaBar, ...
        mergeStruct(struct('M', smcOpts.M, 'returnPath', true), pfOpts));
    sigma2InSample = exp(hF);

    if ismethod(mdl, 'setCovariates') && mdl.nCovariates > 0
        mdl.setCovariates(Z);                            % full series for OOS roll
    end
    fp = inference.forecast.rollingPredictive(mdl, y, res.theta, splitIdx, ...
        struct('J', J, 'M', smcOpts.M, 'pfOpts', pfOpts));

    rec.nu             = nu;
    rec.varForecast    = fp.varForecast;
    rec.logPredDensity = fp.logPredDensity;
    rec.logMarginalLik = res.logMarginalLik;
    rec.sigma2InSample = sigma2InSample;
    rec.stdResid       = standardizeResid(y(1:splitIdx), sigma2InSample, nu);
    rec.theta          = res.theta;
    rec.paramNames     = mdl.paramNames();
    rec.posteriorMean  = mean(res.theta, 1);
    rec.posteriorStd   = std(res.theta, 0, 1);
    if ismethod(mdl, 'omegaPath')
        rec.omegaPath = mdl.omegaPath(thetaBar, hF, y(1:splitIdx));
    else
        rec.omegaPath = [];
    end
end


function rec = emptyModelRecord()
    rec = struct('name', '', 'type', '', 'nu', NaN, ...
                 'varForecast', [], 'logPredDensity', [], 'logMarginalLik', NaN, ...
                 'sigma2InSample', [], 'stdResid', [], 'omegaPath', [], ...
                 'theta', [], 'paramNames', {{}}, 'posteriorMean', [], 'posteriorStd', []);
end


function sc = emptyScore()
    sc = struct('pps', NaN, 'qs1', NaN, 'qs5', NaN, ...
                'mse', NaN, 'mae', NaN, 'r2log', NaN, 'qlike', NaN);
end


function [sc, qlSeries] = scoreModel(varF, lpd, nu, yTest, rvProxy, rvSqrt, alphaQS)
    vHat = sqrt(varF);
    % Standardised innovation quantiles: Gaussian (nu=Inf) -> normal,
    % otherwise the unit-variance scaled-t quantile tinv*sqrt((nu-2)/nu).
    if isinf(nu)
        z1 = norminv(alphaQS(1));
        z2 = norminv(alphaQS(2));
    else
        tScale = sqrt((nu - 2) / nu);
        z1 = tinv(alphaQS(1), nu) * tScale;
        z2 = tinv(alphaQS(2), nu) * tScale;
    end
    sc.pps   = eval.pps(lpd);
    sc.qs1   = eval.quantileScore(yTest, vHat .* z1, alphaQS(1));
    sc.qs5   = eval.quantileScore(yTest, vHat .* z2, alphaQS(2));
    mm       = eval.mseMae(rvSqrt, vHat);
    sc.mse   = mm.mse;
    sc.mae   = mm.mae;
    sc.r2log = eval.r2log(rvProxy, vHat);
    sc.qlike = eval.qlike(varF, rvProxy);
    qlSeries = log(varF) + rvProxy ./ varF;
end


function e = standardizeResid(y, sigma2, nu)
% Normalised residual: Phi^{-1}(F_nu(unit-variance-eps * sqrt(nu/(nu-2)))).
    epsUnit = y ./ sqrt(sigma2);
    if isinf(nu)
        e = epsUnit;
    else
        e = norminv(tcdf(epsUnit .* sqrt(nu / (nu - 2)), nu));
    end
    e(~isfinite(e)) = NaN;
end


function [names, mu, sd] = garchParams(fit, garchType)
% Point estimates as a degenerate posterior (std = NaN; the toolbox object
% does not expose standard errors here). Order: omega, alpha, beta, [gamma], nu.
    names = {'omega', 'alpha', 'beta'};
    mu    = [fit.omega, fit.alpha, fit.beta];
    if strcmp(garchType, 'gjr')
        names{end+1} = 'gamma';
        mu(end+1)    = fit.gamma;
    end
    names{end+1} = 'nu';
    mu(end+1)    = fit.nu;
    sd = nan(size(mu));
end


function [names, mu, sd] = garchRechParams(fit)
% Point estimates for the GARCH-RECH baseline as a degenerate posterior
% (std = NaN). The RNN weights are summarised by beta_0/beta_1; the full
% cell-weight struct lives on fit.cellWeights for interpretability.
    names = {'alpha', 'beta', 'beta0', 'beta1', 'nu'};
    mu    = [fit.alpha, fit.beta, fit.beta0, fit.beta1, fit.nu];
    sd    = nan(size(mu));
end


function descr = describeSeries(y)
    rd = data.preprocess.residualDiagnostics(y);
    st = data.preprocess.stationarityTests(y);
    bd = data.preprocess.bdsTest(y);
    descr = struct( ...
        'n', numel(y), 'mean', mean(y), 'std', std(y), ...
        'skewness', skewness(y), 'kurtosis', kurtosis(y), ...
        'min', min(y), 'max', max(y), ...
        'adf', st.adf, 'pp', st.pp, 'kpss', st.kpss, ...
        'jb', rd.jb, 'archlm', rd.archlm, 'bds', bd);
end


function writeStage4Outputs(bundle, outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    save(fullfile(outDir, 'bundle.mat'), 'bundle');

    names = bundle.comparison.modelNames;
    S     = bundle.comparison.scores;
    rows  = cell(numel(S), 8);
    for m = 1:numel(S)
        s = S(m);
        rows(m, :) = {names{m}, s.pps, s.qs1, s.qs5, s.mse, s.mae, s.r2log, s.qlike};
    end
    tbl = cell2table(rows, 'VariableNames', ...
        {'model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
    writetable(tbl, fullfile(outDir, 'scores.csv'));
end


function [resid, info] = applyMeanEquation(yRaw, splitIdx, cfg)
% Fit the mean equation on the in-sample window and return the full-series
% residuals (fixed-parameter rolling). The Ljung-Box auto-gate decides
% zero-mean vs ARMA(p,q); white-noise returns pass straight through.
% Override via cfg.mean (force/alpha/ljungLag/maxP/maxQ).
    mopts = struct('force', 'auto', 'alpha', 0.05, 'ljungLag', 10, ...
                   'maxP', 3, 'maxQ', 3);
    if isfield(cfg, 'mean'); mopts = mergeStruct(mopts, cfg.mean); end

    meanEq = data.preprocess.meanEquation(yRaw(1:splitIdx), mopts);
    if meanEq.usedARMA && ~isempty(meanEq.model)
        resid = infer(meanEq.model, yRaw);   % full-series 1-step residuals
    else
        resid = yRaw;                        % zero-mean: residual == return
    end

    info = struct('usedARMA', meanEq.usedARMA, 'p', meanEq.p, 'q', meanEq.q, ...
                  'bic', meanEq.bic, 'ljungH', meanEq.ljungH, ...
                  'ljungP', meanEq.ljungP, 'force', mopts.force);
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
