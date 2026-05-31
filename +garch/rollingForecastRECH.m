function out = rollingForecastRECH(fit, y, Z, splitIdx)
% rollingForecastRECH  Fixed-parameter 1-step-ahead GARCH-RECH forecasts.
%
%   out = garch.rollingForecastRECH(fit, y, Z, splitIdx)
%
%   Given a model fitted on the in-sample window (garch.fitGarchRECH), roll
%   the deterministic RECH conditional-variance recursion forward over the
%   whole series with parameters held fixed, and return the genuine
%   1-step-ahead forecasts on the test window y(splitIdx+1:end) together with
%   their log predictive densities. Mirrors garch.rollingForecast for the
%   classic GARCH/GJR baselines.
%
%   Because sigma2_t uses only information through t-1, re-running the
%   recursion from t=1 over the full series reconstructs the in-sample states
%   and then continues into the test window as honest OOS forecasts.
%
%   Inputs
%   ------
%     fit      : struct from garch.fitGarchRECH (alpha, beta, beta0, beta1,
%                cellWeights, cell, dist, nu, seedVar, nCovariates).
%     y        : T-by-1 mean-zero return series (full sample).
%     Z        : T-by-K covariate matrix or [] (must match fit.nCovariates).
%     splitIdx : last in-sample index; the test window is splitIdx+1 .. T.
%
%   Output (struct `out`)
%   ---------------------
%     .sigma2          T-by-1 full conditional-variance path
%     .omegaPath       T-by-1 RNN-driven omega_t path
%     .testIdx         test-window indices (splitIdx+1 .. T)'
%     .sigma2Forecast  1-step-ahead variance forecasts on the test window
%     .logPredDensity  log predictive density of each test return

    arguments
        fit      (1,1) struct
        y        (:,1) double
        Z        (:,:) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
    end

    T = numel(y);
    assert(splitIdx < T, 'rollingForecastRECH:badSplit', ...
        'splitIdx %d must be < series length %d', splitIdx, T);
    K = size(Z, 2);
    assert(K == fit.nCovariates, 'rollingForecastRECH:covShape', ...
        'Z has %d columns; fit expects %d', K, fit.nCovariates);

    P = struct('alpha', fit.alpha, 'beta', fit.beta, ...
               'beta0', fit.beta0, 'beta1', fit.beta1, ...
               'cellWeights', fit.cellWeights);

    [sigma2, omegaPath] = garch.rechCondVar(P, y, Z, fit.cell, fit.seedVar);

    testIdx = (splitIdx + 1 : T)';
    s2Test  = sigma2(testIdx);
    yTest   = y(testIdx);

    if strcmp(fit.dist, 't')
        z   = yTest ./ sqrt(s2Test);
        lpd = -0.5 * log(s2Test) + logStdT(z, fit.nu);
    else
        lpd = -0.5 * log(2 * pi * s2Test) - 0.5 * yTest.^2 ./ s2Test;
    end

    out.sigma2         = sigma2;
    out.omegaPath      = omegaPath;
    out.testIdx        = testIdx;
    out.sigma2Forecast = s2Test;
    out.logPredDensity = lpd;
end


function lp = logStdT(z, nu)
% Log density of a unit-variance standardised Student-t.
    lp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
       - 0.5 * log((nu - 2) * pi) ...
       - ((nu + 1) / 2) * log1p(z.^2 / (nu - 2));
end
