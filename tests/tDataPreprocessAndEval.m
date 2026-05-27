classdef tDataPreprocessAndEval < matlab.unittest.TestCase
% tDataPreprocessAndEval  Coverage for Phase 2 preprocess and eval primitives.

    methods (Test)

        %% ---- standardize ---------------------------------------------------

        function standardizeUsesTrainStats(testCase)
            train = [1; 2; 3; 4; 5];
            test  = [10; 20];
            [trS, teS, stats] = data.preprocess.standardize(train, test);
            testCase.verifyEqual(mean(trS), 0, 'AbsTol', 1e-12);
            testCase.verifyEqual(std(trS),  1, 'AbsTol', 1e-12);
            testCase.verifyEqual(stats.mu, mean(train));
            testCase.verifyEqual(stats.sd, std(train));
            testCase.verifyEqual(teS(1), (10 - stats.mu) / stats.sd, 'AbsTol', 1e-12);
        end

        function standardizeHandlesConstantColumn(testCase)
            train = [ones(10, 1), (1:10)'];
            test  = [ones(3, 1),  (11:13)'];
            [trS, ~, stats] = data.preprocess.standardize(train, test);
            testCase.verifyTrue(stats.constantCol(1));
            testCase.verifyFalse(stats.constantCol(2));
            testCase.verifyEqual(trS(:,1), zeros(10, 1));
        end

        %% ---- trainTestSplit ------------------------------------------------

        function splitChronological(testCase)
            y = (1:100)';
            x = [(1:100)', (101:200)'];
            [yTr, yTe, xTr, ~, idx] = data.preprocess.trainTestSplit(y, x, 0.75);
            testCase.verifyEqual(numel(yTr), 75);
            testCase.verifyEqual(numel(yTe), 25);
            testCase.verifyEqual(yTr(end), 75);
            testCase.verifyEqual(yTe(1),   76);
            testCase.verifyEqual(xTr(end, :), [75, 175]);
            testCase.verifyEqual(idx.trainEnd, 75);
        end

        %% ---- stationarityTests --------------------------------------------

        function stationarityFlagsReturnsButNotPrices(testCase)
            utils.reproducibility(20260516);
            T      = 500;
            mu     = 0.0;
            phi    = 0.95;
            sigma  = 0.4;
            h      = mu + zeros(T, 1);
            for t = 2:T
                h(t) = mu + phi*(h(t-1) - mu) + sigma * randn();
            end
            r = exp(h/2) .* randn(T, 1);
            p = cumsum(r);

            outR = data.preprocess.stationarityTests(r);
            outP = data.preprocess.stationarityTests(p);

            testCase.verifyTrue(outR.stationary, 'returns should be I(0)');
            testCase.verifyFalse(outP.stationary, 'prices should be I(1)');
        end

        %% ---- residualDiagnostics ------------------------------------------

        function diagnosticsCatchArchEffect(testCase)
            utils.reproducibility(20260517);
            T = 800;

            % iid normal -> no ARCH
            iid    = randn(T, 1);
            outIid = data.preprocess.residualDiagnostics(iid);
            testCase.verifyFalse(all(outIid.archlm.h));

            % SV series -> strong ARCH
            mdl   = models.SV();
            y     = mdl.simulate([0, 0.95, 0.4], T);
            outSv = data.preprocess.residualDiagnostics(y);
            testCase.verifyTrue(any(outSv.archlm.h));
        end

        %% ---- Diebold-Mariano -----------------------------------------------

        function dmRejectsWhenLossDiffers(testCase)
            utils.reproducibility(20260518);
            T     = 500;
            loss1 = abs(randn(T, 1)) + 1.0;     % uniformly larger
            loss2 = abs(randn(T, 1));
            out   = eval.dieboldMariano(loss1, loss2);
            testCase.verifyLessThan(out.pValue, 0.05);
            testCase.verifyGreaterThan(out.statistic, 1.96);
        end

        function dmAcceptsWhenLossEqual(testCase)
            utils.reproducibility(20260519);
            T   = 500;
            x   = abs(randn(T, 1));
            out = eval.dieboldMariano(x, x);
            testCase.verifyEqual(out.statistic, 0, 'AbsTol', 1e-9);
            testCase.verifyEqual(out.pValue, 1, 'AbsTol', 1e-9);
        end
    end
end
