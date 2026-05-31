classdef tGarchRECH < matlab.unittest.TestCase
% tGarchRECH  Sanity + smoke tests for the GARCH-RECH deep-learning baseline.
%
%   Needs NO Econometrics Toolbox (RECH is a hand-rolled MLE recursion).
%   fminunc is used if the Optimization Toolbox is present, else fminsearch.
%
%   1. rechReducesToGarchWhenBeta1Zero — beta_1 = 0 collapses RECH to GARCH(1,1).
%   2. recursionPositiveFinite          — all three cells yield positive, finite
%                                          variance paths and are deterministic.
%   3. fitSmokeAllCells                 — srn/lstm/gru fit a short GARCH-t series:
%                                          valid struct, alpha,beta in (0,1),
%                                          alpha+beta<1, nu>2, finite logLik.
%   4. rollingForecastShapes            — OOS roll returns positive test-window
%                                          variances and finite predictive density.

    methods (Static)

        function y = simGarchT(T, omega, alpha, beta, nu, seed)
            rng(seed, 'threefry');
            s2 = zeros(T, 1);
            y  = zeros(T, 1);
            sc = sqrt((nu - 2) / nu);                 % unit-variance t scale
            s2(1) = omega / (1 - alpha - beta);
            y(1)  = sqrt(s2(1)) * sc * trnd(nu);
            for t = 2:T
                s2(t) = omega + alpha * y(t-1)^2 + beta * s2(t-1);
                y(t)  = sqrt(s2(t)) * sc * trnd(nu);
            end
        end
    end

    methods (Test)

        function rechReducesToGarchWhenBeta1Zero(testCase)
            T = 60;
            rng(7, 'threefry');
            y = randn(T, 1);
            P = struct('alpha', 0.10, 'beta', 0.85, 'beta0', 0.05, 'beta1', 0, ...
                       'cellWeights', struct('v', [0 0 0], 'w_h', 0, 'b', 0));
            s2 = garch.rechCondVar(P, y, [], 'srn', var(y));

            manual = zeros(T, 1);
            manual(1) = max(var(y), 1e-8);
            for t = 2:T
                manual(t) = max(0.05 + 0.10 * y(t-1)^2 + 0.85 * manual(t-1), 1e-8);
            end
            testCase.verifyEqual(s2, manual, 'RelTol', 1e-10);
        end

        function recursionPositiveFinite(testCase)
            rng(3, 'threefry');
            T = 80;
            y = 0.5 * randn(T, 1);
            d = 3;                                   % no covariates
            cellsToTest = {'srn', 'lstm', 'gru'};
            for c = 1:numel(cellsToTest)
                ct = cellsToTest{c};
                P  = randomP(ct, d);
                s1 = garch.rechCondVar(P, y, [], ct, var(y));
                s2 = garch.rechCondVar(P, y, [], ct, var(y));   % determinism
                testCase.verifyTrue(all(isfinite(s1)), ct);
                testCase.verifyTrue(all(s1 > 0), ct);
                testCase.verifyEqual(s1, s2, ct);
            end
        end

        function fitSmokeAllCells(testCase)
            y = tGarchRECH.simGarchT(220, 0.05, 0.08, 0.90, 7, 20260516);
            cellsToTest = {'srn', 'lstm', 'gru'};
            for c = 1:numel(cellsToTest)
                ct = cellsToTest{c};
                rng(100 + c, 'threefry');
                fit = garch.fitGarchRECH(y, struct('cell', ct, 'dist', 't', ...
                    'restarts', 2, 'maxEval', 1500));

                testCase.verifyTrue(fit.alpha >= 0 && fit.alpha < 1, ct);
                testCase.verifyTrue(fit.beta  >= 0 && fit.beta  < 1, ct);
                testCase.verifyLessThan(fit.alpha + fit.beta, 1, ct);
                testCase.verifyGreaterThan(fit.nu, 2, ct);
                testCase.verifyTrue(isfinite(fit.logLik), ct);
                testCase.verifyTrue(all(fit.condVar > 0), ct);
                testCase.verifyEqual(numel(fit.condVar), numel(y), ct);
            end
        end

        function rollingForecastShapes(testCase)
            y        = tGarchRECH.simGarchT(240, 0.05, 0.08, 0.90, 7, 99);
            splitIdx = 180;
            rng(5, 'threefry');
            fit = garch.fitGarchRECH(y(1:splitIdx), ...
                struct('cell', 'srn', 'dist', 't', 'restarts', 2, 'maxEval', 1500));
            rf  = garch.rollingForecastRECH(fit, y, [], splitIdx);

            nTest = numel(y) - splitIdx;
            testCase.verifySize(rf.sigma2Forecast, [nTest, 1]);
            testCase.verifySize(rf.logPredDensity, [nTest, 1]);
            testCase.verifyTrue(all(rf.sigma2Forecast > 0));
            testCase.verifyTrue(all(isfinite(rf.logPredDensity)));
        end

    end
end


function P = randomP(cellType, d)
% Small random natural-space parameters for the recursion-only tests.
    P.alpha = 0.08; P.beta = 0.90; P.beta0 = 0.05; P.beta1 = 0.1;
    switch cellType
        case 'srn'
            P.cellWeights = struct('v', 0.05 * randn(1, d), 'w_h', 0.05, 'b', 0.0);
        case 'lstm'
            W = struct();
            for g = {'f','i','o','c'}
                gn = g{1};
                W.(['W_' gn]) = 0.05 * randn(1, d);
                W.(['u_' gn]) = 0.05;
                W.(['b_' gn]) = 0.0;
            end
            P.cellWeights = W;
        case 'gru'
            W = struct();
            for g = {'z','r','h'}
                gn = g{1};
                W.(['W_' gn]) = 0.05 * randn(1, d);
                W.(['u_' gn]) = 0.05;
                W.(['b_' gn]) = 0.0;
            end
            P.cellWeights = W;
    end
end
