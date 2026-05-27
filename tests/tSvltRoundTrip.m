classdef tSvltRoundTrip < matlab.unittest.TestCase
% tSvltRoundTrip  SVLT sanity + small-scale recovery for BOTH leverage
%                 strategies (Cholesky and OCSN).

    properties (TestParameter)
        leverage = {'cholesky', 'ocsn'};
    end

    methods (Test)

        function constructorAcceptsBothStrategies(testCase, leverage)
            mdl = models.SVLT('leverage', leverage);
            testCase.verifyEqual(mdl.leverage, leverage);
        end

        function constructorRejectsUnknownStrategy(testCase)
            testCase.verifyError(@() models.SVLT('leverage', 'foo'), ...
                'MATLAB:validators:mustBeMember');
        end

        function priorSamplesInSupport(testCase, leverage)
            rng(11, 'threefry');
            mdl     = models.SVLT('leverage', leverage);
            samples = mdl.samplePrior(500);

            testCase.verifyTrue(all(abs(samples(:, 2)) < 1));
            testCase.verifyTrue(all(samples(:, 3) > 0));
            testCase.verifyTrue(all(abs(samples(:, 4)) < 1));
            testCase.verifyTrue(all(samples(:, 5) > 2));
        end

        function simulatorProducesFinitePath(testCase, leverage)
            rng(12, 'threefry');
            mdl   = models.SVLT('leverage', leverage);
            theta = [-0.3, 0.97, 0.25, -0.5, 8];
            [y, h] = mdl.simulate(theta, 500);

            testCase.verifySize(y, [500, 1]);
            testCase.verifySize(h, [500, 1]);
            testCase.verifyTrue(all(isfinite(y)));
            testCase.verifyTrue(all(isfinite(h)));
        end

        function leverageProducesNegativeRetVolCorrelation(testCase, leverage)
            % With rho < 0, large negative returns predict elevated
            % next-period log-variance.
            rng(13, 'threefry');
            mdl   = models.SVLT('leverage', leverage);
            theta = [0, 0.95, 0.30, -0.7, 8];
            [y, h] = mdl.simulate(theta, 8000);

            yStd     = y(1:end-1) ./ exp(h(1:end-1) / 2);
            corrYHt1 = corr(yStd, h(2:end));
            testCase.verifyLessThan(corrYHt1, -0.05, ...
                sprintf('leverage=%s: corr(y_std_{t-1}, h_t) = %.3f (expected < 0)', ...
                        leverage, corrYHt1));
        end

        function pfRunsOnSvltData(testCase, leverage)
            rng(14, 'threefry');
            mdl    = models.SVLT('leverage', leverage);
            theta  = [-0.3, 0.95, 0.30, -0.5, 8];
            [y, ~] = mdl.simulate(theta, 300);

            logLik = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 200, 'clipLogWeight', -50));

            testCase.verifyTrue(isfinite(logLik));
            testCase.verifyLessThan(logLik, 0);
        end

        function smallScaleSmcRecoversTruth(testCase, leverage)
            rng(20260516, 'threefry');
            mdl       = models.SVLT('leverage', leverage);
            thetaTrue = [-0.3, 0.95, 0.30, -0.5, 8];
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

            % At this scale rho is poorly identified; require SV-core
            % parameters within 4 posterior SD.
            wellIdentified = [1, 2, 3, 5];
            for k = wellIdentified
                err = abs(postMean(k) - thetaTrue(k));
                testCase.verifyLessThan(err, 4 * postStd(k), ...
                    sprintf('leverage=%s, param %s: err %.2f vs 4*std=%.2f', ...
                            leverage, mdl.paramNames{k}, err, 4 * postStd(k)));
            end
        end

    end
end
