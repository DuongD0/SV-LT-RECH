classdef tMeanEquationAndMCS < matlab.unittest.TestCase
% tMeanEquationAndMCS  Coverage for meanEquation gate + MCS bootstrap.

    methods (Test)

        %% ---- meanEquation -------------------------------------------------

        function zeroMeanOnWhiteNoise(testCase)
            utils.reproducibility(20260520);
            y   = randn(800, 1);
            out = data.preprocess.meanEquation(y);
            testCase.verifyFalse(out.usedARMA, ...
                'Ljung-Box should not reject on i.i.d. normal');
            testCase.verifyEqual(out.residuals, y);
            testCase.verifyEqual(out.p, 0);
            testCase.verifyEqual(out.q, 0);
        end

        function armaFitsAr1Process(testCase)
            utils.reproducibility(20260521);
            T   = 800;
            phi = 0.6;
            y   = zeros(T, 1);
            for t = 2:T
                y(t) = phi * y(t-1) + randn();
            end
            out = data.preprocess.meanEquation(y);
            testCase.verifyTrue(out.usedARMA, ...
                'Ljung-Box must reject on a clear AR(1) process');
            testCase.verifyGreaterThan(out.p + out.q, 0);
            [~, pY] = lbqtest(y, 'Lags', 10);
            [~, pR] = lbqtest(out.residuals, 'Lags', 10);
            testCase.verifyGreaterThan(pR(end), pY(end));
        end

        function forceZeroOverridesGate(testCase)
            utils.reproducibility(20260522);
            T   = 600;
            y   = zeros(T, 1);
            for t = 2:T
                y(t) = 0.5 * y(t-1) + randn();
            end
            out = data.preprocess.meanEquation(y, struct('force', 'zero'));
            testCase.verifyFalse(out.usedARMA);
            testCase.verifyEqual(out.residuals, y);
        end

        %% ---- modelConfidenceSet -------------------------------------------

        function mcsKeepsBestEliminatesWorst(testCase)
            utils.reproducibility(20260523);
            T = 400;
            losses = [randn(T,1), randn(T,1) + 0.2, randn(T,1) + 2.0];
            out = eval.modelConfidenceSet(losses, struct('B', 1000));
            testCase.verifyTrue(out.inSet(1), 'Best model should be in MCS');
            testCase.verifyFalse(out.inSet(3), 'Worst model should be eliminated');
            testCase.verifyGreaterThan(out.eliminationOrder(3), 0);
        end

        function mcsKeepsAllWhenEquallyGood(testCase)
            utils.reproducibility(20260524);
            T = 400;
            base   = randn(T, 1);
            losses = [base + 0.05*randn(T,1), base + 0.05*randn(T,1), base + 0.05*randn(T,1)];
            out = eval.modelConfidenceSet(losses, struct('B', 1000));
            testCase.verifyEqual(sum(out.inSet), 3, ...
                'No model should be eliminated when losses are exchangeable');
        end

        function mcsHandlesTwoModels(testCase)
            utils.reproducibility(20260525);
            T = 300;
            losses = [randn(T,1), randn(T,1) + 3.0];
            out = eval.modelConfidenceSet(losses, struct('B', 1000));
            testCase.verifyTrue(out.inSet(1));
            testCase.verifyFalse(out.inSet(2));
        end
    end
end
