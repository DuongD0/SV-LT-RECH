classdef tSvtRoundTrip < matlab.unittest.TestCase
% tSvtRoundTrip  Sanity + small-scale parameter recovery for SVt.
%
%   1. priorSamplesInSupport          — prior draws satisfy nu > 2, |phi|<1, sigma_eta > 0.
%   2. observationLogLikFinite        — log-density returns finite vectorised over particles.
%   3. observationConvergesToSvAsNuLarge — at nu = 1e5, SVt log p ≈ plain SV log p.
%   4. simulatorHasUnitConditionalVariance — Var(eps_t) ≈ 1 via MC.
%   5. smallScaleSmcRecoversTruth     — SMC posterior median within 3 SD of true.

    methods (Test)

        function priorSamplesInSupport(testCase)
            rng(1, 'threefry');
            mdl     = models.SVt();
            samples = mdl.samplePrior(500);

            testCase.verifyTrue(all(abs(samples(:, 2)) < 1));
            testCase.verifyTrue(all(samples(:, 3) > 0));
            testCase.verifyTrue(all(samples(:, 4) > 2));

            lp = arrayfun(@(i) mdl.logPrior(samples(i, :)), 1:size(samples, 1));
            testCase.verifyTrue(all(isfinite(lp)));
        end

        function observationLogLikFinite(testCase)
            mdl   = models.SVt();
            theta = [0, 0.95, 0.4, 8];
            hT    = randn(100, 1);
            yT    = randn();
            logp  = mdl.observationLogLik(yT, hT, theta);

            testCase.verifySize(logp, [100, 1]);
            testCase.verifyTrue(all(isfinite(logp)));
        end

        function observationConvergesToSvAsNuLarge(testCase)
            % As nu -> infinity, scaled-t -> N(0,1), so SVt log p -> SV log p.
            mdlSv  = models.SV();
            mdlSvt = models.SVt();
            theta  = [0.0, 0.95, 0.4];
            hT     = [-1; 0; 1; 2];
            yT     = 0.7;

            logpSv  = mdlSv.observationLogLik(yT, hT, theta);
            logpSvt = mdlSvt.observationLogLik(yT, hT, [theta, 1e5]);

            testCase.verifyEqual(logpSvt, logpSv, 'AbsTol', 1e-3);
        end

        function simulatorHasUnitConditionalVariance(testCase)
            rng(11, 'threefry');
            mdl   = models.SVt();
            % Pin h_t at mu with phi=0, sigma_eta tiny: y_t = exp(mu/2) * eps_t.
            mu    = 0;
            theta = [mu, 0.0, 1e-6, 8];
            [y, ~] = mdl.simulate(theta, 200000);

            testCase.verifyEqual(var(y), exp(mu), 'RelTol', 0.05);
        end

        function smallScaleSmcRecoversTruth(testCase)
            rng(20260516, 'threefry');
            mdl       = models.SVt();
            thetaTrue = [-0.3, 0.95, 0.30, 8];
            T         = 400;

            [y, ~] = mdl.simulate(thetaTrue, T);

            smcOpts.N             = 200;
            smcOpts.M             = 60;
            smcOpts.nSweeps       = 3;
            smcOpts.proposalScale = 0.4;
            smcOpts.verbose       = false;
            smcOpts.pfOpts        = struct('clipLogWeight', -50);

            result = inference.smc.likelihoodAnneal(mdl, y, smcOpts);

            postMean = mean(result.theta, 1);
            postStd  = std(result.theta, 0, 1);

            for k = 1:numel(thetaTrue)
                err = abs(postMean(k) - thetaTrue(k));
                testCase.verifyLessThan(err, 3 * postStd(k), ...
                    sprintf('Param %s off by %.2f vs 3*std=%.2f', ...
                            mdl.paramNames{k}, err, 3 * postStd(k)));
            end
        end

    end
end
