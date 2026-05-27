function out = rollingPredictive(model, y, thetaPost, splitIdx, opts)
% rollingPredictive  Fixed-parameter rolling 1-step-ahead Bayesian predictive.
%
%   out = rollingPredictive(model, y, thetaPost, splitIdx, opts)
%
%   Holds the in-sample posterior fixed (PROPOSED_METHODOLOGY.md §9.4 sanity
%   option) and rolls a bootstrap PF forward over the full series for each of
%   J thinned posterior particles, then Bayesian-model-averages the per-step
%   predictive densities in log space and the predicted variances linearly.
%
%   Inputs
%   ------
%     model     : a models.Model handle. For covariate models (SVLTRECH) the
%                 caller MUST bind the FULL-series covariates via
%                 model.setCovariates(Z_full) before calling.
%     y         : (T_total)-by-1 full return series (train + test).
%     thetaPost : N-by-K in-sample posterior particles (from likelihoodAnneal).
%     splitIdx  : last in-sample index; test window is splitIdx+1 .. T_total.
%     opts      : struct
%                   .J                 thinned-particle count (default 200)
%                   .M                 PF latent particles (default 200)
%                   .pfOpts            extra opts forwarded to bootstrap
%                   .returnPerParticle keep the J-column matrices (default false)
%
%   Output (struct `out`)
%   ---------------------
%     .testIdx            test-window indices (splitIdx+1 .. T_total)'
%     .logPredDensity     test-window BMA log predictive density
%     .varForecast        test-window BMA one-step predicted variance
%     .logPredDensityAll  full-series BMA log predictive density
%     .varForecastAll     full-series BMA predicted variance
%     .logPredPerParticle (T_total-by-J) only if opts.returnPerParticle

    arguments
        model               models.Model
        y           (:,1)   double
        thetaPost   (:,:)   double
        splitIdx    (1,1)   double {mustBeInteger, mustBePositive}
        opts                struct = struct()
    end

    defaults.J                 = 200;
    defaults.M                 = 200;
    defaults.pfOpts            = struct();
    defaults.returnPerParticle = false;
    opts = mergeStruct(defaults, opts);

    Ttot = numel(y);
    assert(splitIdx < Ttot, 'rollingPredictive:badSplit', ...
        'splitIdx %d must be < series length %d', splitIdx, Ttot);

    N = size(thetaPost, 1);
    J = min(opts.J, N);
    % Even thinning across the posterior particle order (particles are
    % equally weighted after the final anneal step).
    pick      = unique(round(linspace(1, N, J)));
    thetaThin = thetaPost(pick, :);
    J         = size(thetaThin, 1);

    pfOpts                  = mergeStruct(struct('M', opts.M), opts.pfOpts);
    pfOpts.returnIncrements = true;

    logPredAll = zeros(Ttot, J);
    varAll     = zeros(Ttot, J);
    for j = 1:J
        [~, ~, lpd, vf] = inference.pf.bootstrap(model, y, thetaThin(j, :), pfOpts);
        logPredAll(:, j) = lpd;
        varAll(:, j)     = vf;
    end

    % Bayesian model average: log-mean-exp across particles (rowwise, stable).
    m          = max(logPredAll, [], 2);
    logPredBMA = m + log(sum(exp(logPredAll - m), 2)) - log(J);
    varBMA     = mean(varAll, 2);

    testIdx               = (splitIdx + 1 : Ttot)';
    out.testIdx           = testIdx;
    out.logPredDensity    = logPredBMA(testIdx);
    out.varForecast       = varBMA(testIdx);
    out.logPredDensityAll = logPredBMA;
    out.varForecastAll    = varBMA;
    if opts.returnPerParticle
        out.logPredPerParticle = logPredAll;
    end
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
