function out = runStage4(y, Z, splitIdx, cfg, opts)
% runStage4  Stage-4 empirical pipeline: fit, OOS-forecast, score, compare.
%
%   out = experiments.runStage4(y, Z, splitIdx, cfg)
%   out = experiments.runStage4(y, Z, splitIdx, cfg, opts)
%
%   Fits four models on the in-sample window, produces genuine 1-step-ahead
%   OOS forecasts via the fixed-parameter rolling scheme (GARCH via
%   garch.rollingForecast; SV/RECH via inference.forecast.rollingPredictive),
%   scores them on PPS/QS/MSE/MAE/R2LOG/QLIKE with squared returns as the
%   volatility proxy, and compares via Diebold-Mariano (vs the proposed model)
%   and the Model Confidence Set on the per-step QLIKE loss.
%
%   Inputs
%   ------
%     y        : T-by-1 full return series (train + test), mean already removed.
%     Z        : T-by-K covariate matrix (standardised; T-by-0 if none).
%     splitIdx : last in-sample index.
%     cfg      : struct with fields .model (.nCovariates,.leverage), .smc
%                (.N,.M,.nSweeps,.proposalScale,.essThreshold,.targetEss,
%                .verbose), .forecast.J, .eval (.alphaQS,.mcsB),
%                .particleFilter.clipLogWeight, .baseSeed.
%     opts     : struct, optional — .writeOutputs (false), .outDir
%                ('results/stage4').
%
%   Output (struct `out`)
%   ---------------------
%     .modelNames     1-by-4 cellstr
%     .scores         1-by-4 struct array (pps, qs1, qs5, mse, mae, r2log, qlike)
%     .dmVsProposed   1-by-3 struct array (DM of each baseline vs SVLTRECH)
%     .mcs            modelConfidenceSet output over the 4 models
%     .forecasts      struct of per-model test-window varForecast columns
%     .testIdx        test-window indices

    arguments
        y        (:,1) double
        Z        (:,:) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
        cfg      (1,1) struct
        opts           struct = struct()
    end

    defaults.writeOutputs = false;
    defaults.outDir       = fullfile('results', 'stage4');
    opts = mergeStruct(defaults, opts);

    utils.reproducibility(cfg.baseSeed);

    T       = numel(y);
    testIdx = (splitIdx + 1 : T)';
    yTest   = y(testIdx);
    rvProxy = max(yTest .^ 2, 1e-8);    % squared-return variance proxy (floored)
    rvSqrt  = sqrt(rvProxy);
    alphaQS = cfg.eval.alphaQS;

    K = size(Z, 2);

    smcOpts = struct( ...
        'N',             cfg.smc.N, ...
        'M',             cfg.smc.M, ...
        'nSweeps',       cfg.smc.nSweeps, ...
        'proposalScale', cfg.smc.proposalScale, ...
        'essThreshold',  cfg.smc.essThreshold, ...
        'targetEss',     cfg.smc.targetEss, ...
        'verbose',       cfg.smc.verbose, ...
        'pfOpts',        struct('clipLogWeight', cfg.particleFilter.clipLogWeight));
    pfOpts = struct('clipLogWeight', cfg.particleFilter.clipLogWeight);
    J      = cfg.forecast.J;

    modelNames = {'GARCH-t', 'GJR-t', 'SVLT', 'SVLTRECH'};
    nModels    = numel(modelNames);
    scores     = repmat(emptyScore(), 1, nModels);
    qlAll      = zeros(numel(testIdx), nModels);
    fc         = struct();

    %% --- 1. GARCH(1,1)-t ---
    fitG = garch.fitGarch(y(1:splitIdx), struct('type', 'garch', 'dist', 't'));
    rfG  = garch.rollingForecast(fitG, y, splitIdx);
    [scores(1), qlAll(:,1)] = scoreModel(rfG.sigma2Forecast, rfG.logPredDensity, ...
        fitG.nu, yTest, rvProxy, rvSqrt, alphaQS);
    fc.garch = rfG.sigma2Forecast;

    %% --- 2. GJR-GARCH(1,1)-t ---
    fitGJR = garch.fitGarch(y(1:splitIdx), struct('type', 'gjr', 'dist', 't'));
    rfGJR  = garch.rollingForecast(fitGJR, y, splitIdx);
    [scores(2), qlAll(:,2)] = scoreModel(rfGJR.sigma2Forecast, rfGJR.logPredDensity, ...
        fitGJR.nu, yTest, rvProxy, rvSqrt, alphaQS);
    fc.gjr = rfGJR.sigma2Forecast;

    %% --- 3. SVLT (SV baseline) ---
    mdlSVLT = models.SVLT();                        % default cholesky leverage
    resSVLT = inference.smc.likelihoodAnneal(mdlSVLT, y(1:splitIdx), smcOpts);
    nuSVLT  = mean(resSVLT.theta(:, 5));            % nu is index 5
    fpSVLT  = inference.forecast.rollingPredictive(mdlSVLT, y, resSVLT.theta, ...
        splitIdx, struct('J', J, 'M', cfg.smc.M, 'pfOpts', pfOpts));
    [scores(3), qlAll(:,3)] = scoreModel(fpSVLT.varForecast, fpSVLT.logPredDensity, ...
        nuSVLT, yTest, rvProxy, rvSqrt, alphaQS);
    fc.svlt = fpSVLT.varForecast;

    %% --- 4. SVLTRECH (proposed, SRN) ---
    mdlRECH = models.SVLTRECH('nCovariates', K, 'leverage', cfg.model.leverage);
    mdlRECH.setCovariates(Z(1:splitIdx, :));        % in-sample covariates for fitting
    resRECH = inference.smc.likelihoodAnneal(mdlRECH, y(1:splitIdx), smcOpts);
    nuRECH  = mean(resRECH.theta(:, 5));
    mdlRECH.setCovariates(Z);                       % FULL covariates for rolling OOS
    fpRECH  = inference.forecast.rollingPredictive(mdlRECH, y, resRECH.theta, ...
        splitIdx, struct('J', J, 'M', cfg.smc.M, 'pfOpts', pfOpts));
    [scores(4), qlAll(:,4)] = scoreModel(fpRECH.varForecast, fpRECH.logPredDensity, ...
        nuRECH, yTest, rvProxy, rvSqrt, alphaQS);
    fc.svltrech = fpRECH.varForecast;

    %% --- Diebold-Mariano: each baseline vs the proposed model (col 4) ---
    dmVsProposed = repmat(eval.dieboldMariano(qlAll(:,1), qlAll(:,4), 1), 1, nModels - 1);
    for m = 2:(nModels - 1)
        dmVsProposed(m) = eval.dieboldMariano(qlAll(:, m), qlAll(:, nModels), 1);
    end

    %% --- Model Confidence Set on per-step QLIKE loss ---
    mcs = eval.modelConfidenceSet(qlAll, struct('B', cfg.eval.mcsB));

    out.modelNames   = modelNames;
    out.scores       = scores;
    out.dmVsProposed = dmVsProposed;
    out.mcs          = mcs;
    out.forecasts    = fc;
    out.testIdx      = testIdx;

    if opts.writeOutputs
        writeStage4Outputs(out, opts.outDir);
    end
end


function sc = emptyScore()
% Fixed field order so repmat'd struct array is homogeneous.
    sc = struct('pps', NaN, 'qs1', NaN, 'qs5', NaN, ...
                'mse', NaN, 'mae', NaN, 'r2log', NaN, 'qlike', NaN);
end


function [sc, qlSeries] = scoreModel(varF, lpd, nu, yTest, rvProxy, rvSqrt, alphaQS)
% scoreModel  Five predictive scores + per-obs QLIKE for one model.
%   varF : test-window one-step variance forecasts (positive).
%   lpd  : test-window log predictive densities.
%   nu   : Student-t dof (for the quantile forecast).
    vHat   = sqrt(varF);
    tScale = sqrt((nu - 2) / nu);                  % unit-variance t scaling
    sc.pps   = eval.pps(lpd);
    sc.qs1   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(1), nu) .* tScale, alphaQS(1));
    sc.qs5   = eval.quantileScore(yTest, vHat .* tinv(alphaQS(2), nu) .* tScale, alphaQS(2));
    mm       = eval.mseMae(rvSqrt, vHat);
    sc.mse   = mm.mse;
    sc.mae   = mm.mae;
    sc.r2log = eval.r2log(rvProxy, vHat);
    sc.qlike = eval.qlike(varF, rvProxy);
    qlSeries = log(varF) + rvProxy ./ varF;        % per-obs QLIKE for DM / MCS
end


function writeStage4Outputs(out, outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    rows = cell(numel(out.scores), 8);
    for m = 1:numel(out.scores)
        s = out.scores(m);
        rows(m, :) = {out.modelNames{m}, s.pps, s.qs1, s.qs5, s.mse, s.mae, s.r2log, s.qlike};
    end
    tbl = cell2table(rows, 'VariableNames', ...
        {'model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
    writetable(tbl, fullfile(outDir, 'scores.csv'));

    fcTbl = table(out.testIdx, out.forecasts.garch, out.forecasts.gjr, ...
                  out.forecasts.svlt, out.forecasts.svltrech, ...
        'VariableNames', {'testIdx','GARCH','GJR','SVLT','SVLTRECH'});
    writetable(fcTbl, fullfile(outDir, 'forecasts.csv'));
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
