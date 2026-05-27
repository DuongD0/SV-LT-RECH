classdef tRollingPredictive < matlab.unittest.TestCase
% tRollingPredictive  Fixed-parameter rolling 1-step-ahead predictive.

    methods (Test)

        function shapesAndPositivity(testCase)
            rng(11, 'threefry');
            mdl    = models.SV();
            theta  = [-0.2, 0.95, 0.30];
            [y, ~] = mdl.simulate(theta, 400);
            splitIdx = 300;

            % A small "posterior": jittered copies of the truth.
            N = 40;
            thetaPost = theta + 0.02 * randn(N, 3);
            thetaPost(:,2) = min(max(thetaPost(:,2), -0.99), 0.99);   % phi in (-1,1)
            thetaPost(:,3) = abs(thetaPost(:,3));                     % sigma_eta > 0

            out = inference.forecast.rollingPredictive(mdl, y, thetaPost, ...
                splitIdx, struct('J', 20, 'M', 200));

            nTest = numel(y) - splitIdx;
            testCase.verifySize(out.logPredDensity, [nTest, 1]);
            testCase.verifySize(out.varForecast,    [nTest, 1]);
            testCase.verifyEqual(out.testIdx, (splitIdx+1:numel(y))');
            testCase.verifyTrue(all(isfinite(out.logPredDensity)));
            testCase.verifyTrue(all(out.varForecast > 0));
        end

        function bmaWithinPerParticleRange(testCase)
            % log-mean-exp over particles must lie within [min, max] of the
            % per-particle predictive log-densities at every t.
            rng(12, 'threefry');
            mdl    = models.SV();
            theta  = [0.0, 0.92, 0.35];
            [y, ~] = mdl.simulate(theta, 250);
            splitIdx = 180;
            thetaPost = repmat(theta, 8, 1) + 0.01 * randn(8, 3);
            thetaPost(:,3) = abs(thetaPost(:,3));

            out = inference.forecast.rollingPredictive(mdl, y, thetaPost, ...
                splitIdx, struct('J', 8, 'M', 200, 'returnPerParticle', true));

            lo  = min(out.logPredPerParticle, [], 2);
            hi  = max(out.logPredPerParticle, [], 2);
            bma = out.logPredDensityAll;     % full-series BMA path
            testCase.verifyTrue(all(bma >= lo - 1e-9));
            testCase.verifyTrue(all(bma <= hi + 1e-9));
        end
    end
end
