classdef tSvmRoundTrip < matlab.unittest.TestCase
% tSvmRoundTrip  Sanity + small-scale parameter recovery for SV-in-mean.
%
%   1. priorSamplesInSupport         — prior draws satisfy |phi|<1, sigma_eta>0,
%                                       and yield finite log-prior.
%   2. observationLogLikFinite       — log-density vectorised over particles.
%   3. observationPeaksAtInMeanMean  — argmax over y of the obs log-density
%                                       equals alpha0 + lambda*exp(h).
%   4. simulatorInMeanShift          — with variance pinned, sample mean
%                                       tracks alpha0 + lambda*exp(mu).
%   5. zeroLambdaMatchesPlainSv      — at alpha0=0, lambda=0 the SVM obs
%                                       density equals plain SV.
%   6. smallScaleSmcRecoversTruth    — SMC posterior mean within 3 SD of true.

    methods (Test)

        function priorSamplesInSupport(testCase)
            rng(1, 'threefry');
            mdl     = models.SVM();
            samples = mdl.samplePrior(500);

            testCase.verifyTrue(all(abs(samples(:, 2)) < 1));
            testCase.verifyTrue(all(samples(:, 3) > 0));

            lp = arrayfun(@(i) mdl.logPrior(samples(i, :)), 1:size(samples, 1));
            testCase.verifyTrue(all(isfinite(lp)));
        end

        function observationLogLikFinite(testCase)
            mdl   = models.SVM();
            theta = [0, 0.95, 0.4, 0.2, 0.3];
            hT    = randn(100, 1);
            yT    = randn();
            logp  = mdl.observationLogLik(yT, hT, theta);

            testCase.verifySize(logp, [100, 1]);
            testCase.verifyTrue(all(isfinite(logp)));
        end

        function observationPeaksAtInMeanMean(testCase)
            % For fixed h, the Gaussian obs density is maximised at the
            % conditional mean alpha0 + lambda*exp(h).
            mdl    = models.SVM();
            alpha0 = 0.5; lambda = 0.3; h = 0.2;
            theta  = [0, 0.9, 0.3, alpha0, lambda];
            ygrid  = linspace(-5, 5, 4001)';
            lp     = arrayfun(@(y) mdl.observationLogLik(y, h, theta), ygrid);
            [~, ix] = max(lp);
            expectedMean = alpha0 + lambda * exp(h);
            testCase.verifyEqual(ygrid(ix), expectedMean, 'AbsTol', 5e-3);
        end

        function simulatorInMeanShift(testCase)
            rng(11, 'threefry');
            mdl    = models.SVM();
            mu     = 0; alpha0 = 0.5; lambda = 0.3;
            theta  = [mu, 0.0, 1e-6, alpha0, lambda];   % variance pinned ~ exp(mu)
            [y, ~] = mdl.simulate(theta, 200000);
            expectedMean = alpha0 + lambda * exp(mu);
            testCase.verifyEqual(mean(y), expectedMean, 'AbsTol', 0.02);
            testCase.verifyEqual(var(y), exp(mu), 'RelTol', 0.05);
        end

        function zeroLambdaMatchesPlainSv(testCase)
            % alpha0=0, lambda=0 -> SVM observation density == plain SV.
            mdlSv  = models.SV();
            mdlSvm = models.SVM();
            theta  = [0.0, 0.95, 0.4];
            hT     = [-1; 0; 1; 2];
            yT     = 0.7;
            logpSv  = mdlSv.observationLogLik(yT, hT, theta);
            logpSvm = mdlSvm.observationLogLik(yT, hT, [theta, 0, 0]);
            testCase.verifyEqual(logpSvm, logpSv, 'AbsTol', 1e-10);
        end

        function smallScaleSmcRecoversTruth(testCase)
            rng(20260516, 'threefry');
            mdl       = models.SVM();
            thetaTrue = [-0.3, 0.95, 0.30, 0.10, 0.20];
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
