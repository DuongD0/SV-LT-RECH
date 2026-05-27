classdef tPriorPredictive < matlab.unittest.TestCase
% tPriorPredictive  Likelihood direction sanity check.
%
%   Verifies the posterior moves in the data-implied direction. Two
%   scenarios:
%     (a) low-volatility data  -> posterior mu should be NEGATIVE
%                                 (log-variance is small)
%     (b) high-volatility data -> posterior mu should be POSITIVE
%
%   A flipped likelihood sign or an inverted annealing schedule would
%   reverse these directions. We also verify that every posterior
%   particle is in support (finite logPrior) and the log marginal
%   likelihood is finite.

    methods (Test)

        function lowVolDataDrivesMuNegative(testCase)
            utils.reproducibility(20260516);

            mdl  = models.SV();
            T    = 200;
            y    = randn(T, 1) * 0.05;          % sd 0.05 -> h ~ 2*log(0.05) = -6
            opts = struct('N', 800, 'M', 100, 'nSweeps', 3, ...
                          'verbose', false, 'proposalScale', 0.3);
            res  = inference.smc.likelihoodAnneal(mdl, y, opts);

            postMean = mean(res.theta);
            testCase.verifyLessThan(postMean(1), -2.0, ...
                'mu posterior should be strongly negative for low-vol data');
            testCase.verifyTrue(isfinite(res.logMarginalLik));
        end

        function highVolDataDrivesMuPositive(testCase)
            utils.reproducibility(20260517);

            mdl  = models.SV();
            T    = 200;
            y    = randn(T, 1) * 3.0;           % sd 3 -> h ~ 2*log(3) = 2.2
            opts = struct('N', 800, 'M', 100, 'nSweeps', 3, ...
                          'verbose', false, 'proposalScale', 0.3);
            res  = inference.smc.likelihoodAnneal(mdl, y, opts);

            postMean = mean(res.theta);
            testCase.verifyGreaterThan(postMean(1), 1.0, ...
                'mu posterior should be positive for high-vol data');
            testCase.verifyTrue(isfinite(res.logMarginalLik));
        end

        function allParticlesInSupport(testCase)
            utils.reproducibility(20260518);

            mdl  = models.SV();
            T    = 100;
            y    = randn(T, 1);
            opts = struct('N', 500, 'M', 80, 'nSweeps', 2, ...
                          'verbose', false, 'proposalScale', 0.3);
            res  = inference.smc.likelihoodAnneal(mdl, y, opts);

            for i = 1:size(res.theta, 1)
                testCase.verifyTrue(isfinite(mdl.logPrior(res.theta(i, :))), ...
                    sprintf('Particle %d out of support after SMC', i));
            end
        end
    end
end
