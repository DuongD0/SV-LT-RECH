classdef tGarchBaselines < matlab.unittest.TestCase
% tGarchBaselines  GARCH/GJR baseline fit + rolling-forecast smoke.
%
%   REQUIRES the Econometrics Toolbox. Each test assumes the toolbox is
%   present; on environments without it (e.g. the Colab Stats-only bundle)
%   the tests are filtered out rather than failed.

    methods (TestMethodSetup)
        function requireEcon(testCase)
            testCase.assumeTrue(~isempty(ver('econ')), ...
                'Econometrics Toolbox not available; skipping GARCH baselines.');
        end
    end

    methods (Test)

        function garchTFitIsStationaryAndSane(testCase)
            rng(101, 'threefry');
            trueMdl = garch('Constant', 0.02, 'GARCH', 0.90, 'ARCH', 0.08, ...
                            'Distribution', struct('Name', 't', 'DoF', 8));
            [~, y] = simulate(trueMdl, 4000);   % 2nd output = returns (1st = variance)

            fit = garch.fitGarch(y, struct('type', 'garch', 'dist', 't'));

            testCase.verifyGreaterThan(fit.omega, 0);
            testCase.verifyGreaterThanOrEqual(fit.alpha, 0);
            testCase.verifyGreaterThanOrEqual(fit.beta, 0);
            testCase.verifyLessThan(fit.alpha + fit.beta, 1);   % stationarity
            testCase.verifyGreaterThan(fit.nu, 2);
            testCase.verifyTrue(isfinite(fit.logLik));
            testCase.verifyEqual(fit.nParams, 4);               % omega,alpha,beta,nu
            testCase.verifySize(fit.condVar, [4000, 1]);
            testCase.verifyTrue(all(fit.condVar > 0));
        end

        function garchTRecoversPersistenceLoosely(testCase)
            rng(102, 'threefry');
            trueMdl = garch('Constant', 0.02, 'GARCH', 0.90, 'ARCH', 0.08, ...
                            'Distribution', struct('Name', 't', 'DoF', 8));
            [~, y] = simulate(trueMdl, 5000);   % 2nd output = returns (1st = variance)
            fit = garch.fitGarch(y, struct('type', 'garch', 'dist', 't'));

            % Generous bands: estimation noise at T=5000 is non-trivial.
            testCase.verifyGreaterThan(fit.beta, 0.75);
            testCase.verifyLessThan(fit.beta, 0.99);
            testCase.verifyGreaterThan(fit.alpha, 0.005);
            testCase.verifyLessThan(fit.alpha, 0.25);
        end

        function gjrFitHasLeverageCoefficient(testCase)
            rng(103, 'threefry');
            trueMdl = gjr('Constant', 0.02, 'GARCH', 0.88, 'ARCH', 0.04, ...
                          'Leverage', 0.10, ...
                          'Distribution', struct('Name', 't', 'DoF', 8));
            [~, y] = simulate(trueMdl, 5000);   % 2nd output = returns (1st = variance)
            fit = garch.fitGarch(y, struct('type', 'gjr', 'dist', 't'));

            testCase.verifyNotEmpty(fit.gamma);
            testCase.verifyTrue(isfinite(fit.gamma));
            testCase.verifyEqual(fit.nParams, 5);   % omega,alpha,beta,gamma,nu
        end

        function rollingForecastIsWellShapedAndScores(testCase)
            rng(104, 'threefry');
            trueMdl = garch('Constant', 0.02, 'GARCH', 0.90, 'ARCH', 0.08, ...
                            'Distribution', struct('Name', 't', 'DoF', 8));
            [~, y]   = simulate(trueMdl, 2500);   % 2nd output = returns (1st = variance)
            splitIdx = 2000;

            fit = garch.fitGarch(y(1:splitIdx), struct('type', 'garch', 'dist', 't'));
            out = garch.rollingForecast(fit, y, splitIdx);

            nTest = numel(y) - splitIdx;
            testCase.verifySize(out.sigma2Forecast, [nTest, 1]);
            testCase.verifySize(out.logPredDensity, [nTest, 1]);
            testCase.verifyTrue(all(out.sigma2Forecast > 0));
            testCase.verifyTrue(all(isfinite(out.logPredDensity)));

            % Forecasts must feed the +eval losses without error.
            rvTest = y(out.testIdx).^2;   % squared returns as the volatility proxy
            ql  = eval.qlike(out.sigma2Forecast, rvTest);
            pp  = eval.pps(out.logPredDensity);
            testCase.verifyTrue(all(isfinite(ql(:))));
            testCase.verifyTrue(all(isfinite(pp(:))));
        end

    end
end
