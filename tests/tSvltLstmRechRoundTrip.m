classdef tSvltLstmRechRoundTrip < matlab.unittest.TestCase
% tSvltLstmRechRoundTrip  SV-LT-LSTM-RECH (§8.4) sanity + small-scale SMC
%   smoke for BOTH leverage strategies. Scoped to "code path runs and
%   produces well-shaped, finite output", not convergence (cf. Phase 3
%   smoke philosophy in docs/PHASE_3_SCAFFOLD_COMPLETE.md §3).

    properties (TestParameter)
        leverage = {'cholesky', 'ocsn'};
    end

    methods (Static)
        function n = thetaLen(K)
            n = 15 + 4 * (3 + K);   % 7 core + 4 gates x (nInputs + u + b)
        end

        function theta = makeTheta(mdl)
            % Deterministic in-support theta: core values + small gate weights.
            K       = mdl.nCovariates;
            nGateW  = 4 * (3 + K + 2);
            core    = [-0.5, 0.97, 0.25, -0.5, 8, 0.05, 0.20];
            gateW   = 0.05 * ones(1, nGateW);
            theta   = [core, gateW];
        end
    end

    methods (Test)

        function constructorAcceptsBothStrategies(testCase, leverage)
            mdl = models.SVLTLSTMRECH('leverage', leverage);
            testCase.verifyEqual(mdl.leverage, leverage);
        end

        function paramNamesLengthMatchesTheta(testCase)
            for K = [0, 2]
                mdl = models.SVLTLSTMRECH('nCovariates', K);
                expected = tSvltLstmRechRoundTrip.thetaLen(K);
                testCase.verifyNumElements(mdl.paramNames(), expected);
                testCase.verifySize(mdl.samplePrior(3), [3, expected]);
            end
        end

        function priorSamplesInSupport(testCase, leverage)
            rng(31, 'threefry');
            mdl     = models.SVLTLSTMRECH('leverage', leverage);
            samples = mdl.samplePrior(500);

            testCase.verifyTrue(all(abs(samples(:, 2)) < 1));   % phi
            testCase.verifyTrue(all(samples(:, 3) > 0));         % sigma_eta
            testCase.verifyTrue(all(abs(samples(:, 4)) < 1));   % rho
            testCase.verifyTrue(all(samples(:, 5) > 2));         % nu
            testCase.verifyTrue(all(samples(:, 6) >= 0 & samples(:, 6) <= 0.5)); % beta_0
            testCase.verifyTrue(all(samples(:, 7) >= 0 & samples(:, 7) <= 0.5)); % beta_1
        end

        function logPriorFiniteInSupportAndRejectsOutside(testCase)
            mdl   = models.SVLTLSTMRECH();
            theta = tSvltLstmRechRoundTrip.makeTheta(mdl);
            testCase.verifyTrue(isfinite(mdl.logPrior(theta)));

            bad = theta; bad(2) = 1.5;   % phi out of (-1, 1)
            testCase.verifyEqual(mdl.logPrior(bad), -Inf);
        end

        function simulatorProducesFinitePath(testCase, leverage)
            rng(32, 'threefry');
            mdl    = models.SVLTLSTMRECH('leverage', leverage);
            theta  = tSvltLstmRechRoundTrip.makeTheta(mdl);
            [y, h] = mdl.simulate(theta, 500);

            testCase.verifySize(y, [500, 1]);
            testCase.verifySize(h, [500, 1]);
            testCase.verifyTrue(all(isfinite(y)));
            testCase.verifyTrue(all(isfinite(h)));
        end

        function pfRunsOnLstmData(testCase, leverage)
            rng(33, 'threefry');
            mdl    = models.SVLTLSTMRECH('leverage', leverage);
            theta  = tSvltLstmRechRoundTrip.makeTheta(mdl);
            [y, ~] = mdl.simulate(theta, 300);

            logLik = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 200, 'clipLogWeight', -50));

            testCase.verifyTrue(isfinite(logLik));
            testCase.verifyLessThan(logLik, 0);
        end

        function covariatePathRunsThroughPf(testCase)
            % nCovariates > 0 exercises the z-input branch end-to-end.
            rng(34, 'threefry');
            K   = 2;
            T   = 200;
            Z   = randn(T, K);
            mdl = models.SVLTLSTMRECH('nCovariates', K, 'covariates', Z);
            theta = tSvltLstmRechRoundTrip.makeTheta(mdl);

            [y, ~] = mdl.simulate(theta, T);
            logLik = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 150, 'clipLogWeight', -50));

            testCase.verifyTrue(all(isfinite(y)));
            testCase.verifyTrue(isfinite(logLik));
        end

        function smallScaleSmcRunsAndIsWellShaped(testCase, leverage)
            rng(20260516, 'threefry');
            mdl       = models.SVLTLSTMRECH('leverage', leverage);
            thetaTrue = tSvltLstmRechRoundTrip.makeTheta(mdl);
            T         = 300;
            [y, ~]    = mdl.simulate(thetaTrue, T);

            smcOpts.N             = 150;
            smcOpts.M             = 60;
            smcOpts.nSweeps       = 3;
            smcOpts.proposalScale = 0.4;
            smcOpts.verbose       = false;
            smcOpts.pfOpts        = struct('clipLogWeight', -50);

            result = inference.smc.likelihoodAnneal(mdl, y, smcOpts);

            expected = tSvltLstmRechRoundTrip.thetaLen(0);
            testCase.verifySize(result.theta, [150, expected]);
            testCase.verifyTrue(all(isfinite(result.theta(:))));
            testCase.verifyTrue(all(std(result.theta, 0, 1) > 0));
            testCase.verifyTrue(isfinite(result.logMarginalLik));
        end

    end
end
