function out = rollingForecast(fit, y, splitIdx)
% rollingForecast  Fixed-parameter 1-step-ahead variance forecasts.
%
%   out = garch.rollingForecast(fit, y, splitIdx)
%
%   Given a model fitted on the in-sample window (garch.fitGarch), roll the
%   GARCH(1,1) / GJR(1,1) conditional-variance recursion forward over the
%   whole series with the estimated parameters held fixed, and return the
%   genuine 1-step-ahead forecasts on the test window y(splitIdx+1:end)
%   together with their log predictive densities.
%
%   This is the standard fixed-parameter rolling-filter OOS scheme: the
%   variance recursion at time t uses only information through t-1, so
%   sigma2(t) is the 1-step-ahead forecast issued at t-1. (Cheaper than
%   refitting every step; refit-every-100 per §9.4 is a future option.)
%
%   Inputs
%   ------
%     fit      : struct from garch.fitGarch (needs omega, alpha, beta,
%                gamma, nu, dist).
%     y        : T-by-1 mean-zero return series (full sample).
%     splitIdx : last in-sample index; the test window is splitIdx+1 .. T.
%
%   Output (struct `out`)
%   ---------------------
%     .sigma2          T-by-1 full conditional-variance path
%     .testIdx         test-window indices (splitIdx+1 .. T)'
%     .sigma2Forecast  1-step-ahead variance forecasts on the test window
%     .logPredDensity  log predictive density of each test return
%
%   Feeds the +eval losses directly: qlike(sigma2Forecast, rvTest),
%   mseMae(sqrt(rvTest), sqrt(sigma2Forecast)), pps(logPredDensity).

    arguments
        fit      (1,1) struct
        y        (:,1) double
        splitIdx (1,1) double {mustBeInteger, mustBePositive}
    end

    T = numel(y);
    assert(splitIdx < T, 'rollingForecast:badSplit', ...
        'splitIdx %d must be < series length %d', splitIdx, T);

    omega = fit.omega;
    alpha = scalarCoef(fit.alpha, 'alpha');
    beta  = scalarCoef(fit.beta,  'beta');
    if isempty(fit.gamma)
        gamma = 0;
    else
        gamma = scalarCoef(fit.gamma, 'gamma');
    end

    sigma2    = zeros(T, 1);
    sigma2(1) = var(y(1:splitIdx));        % unconditional seed from in-sample
    for t = 2:T
        rl        = y(t - 1);
        levTerm   = gamma * rl^2 * (rl < 0);
        sigma2(t) = omega + alpha * rl^2 + levTerm + beta * sigma2(t - 1);
    end

    testIdx = (splitIdx + 1 : T)';
    s2Test  = sigma2(testIdx);
    yTest   = y(testIdx);

    if strcmp(fit.dist, 't')
        z   = yTest ./ sqrt(s2Test);
        nu  = fit.nu;
        lpd = -0.5 * log(s2Test) + logStdT(z, nu);
    else
        lpd = -0.5 * log(2 * pi * s2Test) - 0.5 * yTest.^2 ./ s2Test;
    end

    out.sigma2         = sigma2;
    out.testIdx        = testIdx;
    out.sigma2Forecast = s2Test;
    out.logPredDensity = lpd;
end


function c = scalarCoef(v, name)
    assert(isscalar(v), 'rollingForecast:nonScalarCoef', ...
        'rollingForecast supports order (1,1) only; %s has %d elements', ...
        name, numel(v));
    c = v;
end


function lp = logStdT(z, nu)
% Log density of a unit-variance standardized Student-t (matches the SV
% models' observation density): the variable z*sqrt(nu/(nu-2)) ~ t_nu.
    lp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
       - 0.5 * log((nu - 2) * pi) ...
       - ((nu + 1) / 2) * log1p(z.^2 / (nu - 2));
end
