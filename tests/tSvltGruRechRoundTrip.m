classdef tSvltGruRechRoundTrip < matlab.unittest.TestCase
% tSvltGruRechRoundTrip  SV-LT-GRU-RECH (§8.5) sanity + small-scale SMC
%   smoke for BOTH leverage strategies. Scoped to "code path runs and
%   produces well-shaped, finite output", not convergence (cf. Phase 3
%   smoke philosophy in docs/PHASE_3_SCAFFOLD_COMPLETE.md §3).

    properties (TestParameter)
        leverage = {'cholesky', 'ocsn'};
    end

    methods (Static)
        function n = thetaLen(K)
            n = 13 + 3 * (3 + K);   % 7 core + 3 gates x (nInputs + u + b)
        end

        function theta = makeTheta(mdl)
            K       = mdl.nCovariates;
            nGateW  = 3 * (3 + K + 2);
            core    = [-0.5, 0.97, 0.25, -0.5, 8, 0.05, 0.20];
            gateW   = 0.05 * ones(1, nGateW);
            theta   = [core, gateW];
        end
    end

    methods (Test)

        function constructorAcceptsBothStrategies(testCase, leverage)
            mdl = models.SVLTGRURECH('leverage', leverage);
            testCase.verifyEqual(mdl.leverage, leverage);
        end

        function paramNamesLengthMatchesTheta(testCase)
            for K = [0, 2]
                mdl = models.SVLTGRURECH('nCovariates', K);
                expected = tSvltGruRechRoundTrip.thetaLen(K);
                testCase.verifyNumElements(mdl.paramNames(), expected);
                testCase.verifySize(mdl.samplePrior(3), [3, expected]);
            end
        end

        function priorSamplesInSupport(testCase, leverage)
            rng(41, 'threefry');
            mdl     = models.SVLTGRURECH('leverage', leverage);
            samples = mdl.samplePrior(500);

            testCase.verifyTrue(all(abs(samples(:, 2)) < 1));   % phi
            testCase.verifyTrue(all(samples(:, 3) > 0));         % sigma_eta
            testCase.verifyTrue(all(abs(samples(:, 4)) < 1));   % rho
            testCase.verifyTrue(all(samples(:, 5) > 2));         % nu
            testCase.verifyTrue(all(samples(:, 6) >= 0 & samples(:, 6) <= 0.5)); % beta_0
            testCase.verifyTrue(all(samples(:, 7) >= 0 & samples(:, 7) <= 0.5)); % beta_1
        end

        function logPriorFiniteInSupportAndRejectsOutside(testCase)
            mdl   = models.SVLTGRURECH();
            theta = tSvltGruRechRoundTrip.makeTheta(mdl);
            testCase.verifyTrue(isfinite(mdl.logPrior(theta)));

            bad = theta; bad(2) = 1.5;   % phi out of (-1, 1)
            testCase.verifyEqual(mdl.logPrior(bad), -Inf);
        end

        function simulatorProducesFinitePath(testCase, leverage)
            rng(42, 'threefry');
            mdl    = models.SVLTGRURECH('leverage', leverage);
            theta  = tSvltGruRechRoundTrip.makeTheta(mdl);
            [y, h] = mdl.simulate(theta, 500);

            testCase.verifySize(y, [500, 1]);
            testCase.verifySize(h, [500, 1]);
            testCase.verifyTrue(all(isfinite(y)));
            testCase.verifyTrue(all(isfinite(h)));
        end

        function pfRunsOnGruData(testCase, leverage)
            rng(43, 'threefry');
            mdl    = models.SVLTGRURECH('leverage', leverage);
            theta  = tSvltGruRechRoundTrip.makeTheta(mdl);
            [y, ~] = mdl.simulate(theta, 300);

            logLik = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 200, 'clipLogWeight', -50));

            testCase.verifyTrue(isfinite(logLik));
            testCase.verifyLessThan(logLik, 0);
        end

        function covariatePathRunsThroughPf(testCase)
            rng(44, 'threefry');
            K   = 2;
            T   = 200;
            Z   = randn(T, K);
            mdl = models.SVLTGRURECH('nCovariates', K, 'covariates', Z);
            theta = tSvltGruRechRoundTrip.makeTheta(mdl);

            [y, ~] = mdl.simulate(theta, T);
            logLik = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 150, 'clipLogWeight', -50));

            testCase.verifyTrue(all(isfinite(y)));
            testCase.verifyTrue(isfinite(logLik));
        end

        function smallScaleSmcRunsAndIsWellShaped(testCase, leverage)
            rng(20260516, 'threefry');
            mdl       = models.SVLTGRURECH('leverage', leverage);
            thetaTrue = tSvltGruRechRoundTrip.makeTheta(mdl);
            T         = 300;
            [y, ~]    = mdl.simulate(thetaTrue, T);

            smcOpts.N             = 150;
            smcOpts.M             = 60;
            smcOpts.nSweeps       = 3;
            smcOpts.proposalScale = 0.4;
            smcOpts.verbose       = false;
            smcOpts.pfOpts        = struct('clipLogWeight', -50);

            result = inference.smc.likelihoodAnneal(mdl, y, smcOpts);

            expected = tSvltGruRechRoundTrip.thetaLen(0);
            testCase.verifySize(result.theta, [150, expected]);
            testCase.verifyTrue(all(isfinite(result.theta(:))));
            testCase.verifyTrue(all(std(result.theta, 0, 1) > 0));
            testCase.verifyTrue(isfinite(result.logMarginalLik));
        end

    end
end
