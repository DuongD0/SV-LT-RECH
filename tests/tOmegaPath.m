classdef tOmegaPath < matlab.unittest.TestCase
% tOmegaPath  SVLTRECH.omegaPath shape + first value (= beta_0).

    methods (Test)
        function shapeAndFirstValue(testCase)
            rng(5, 'threefry');
            mdl   = models.SVLTRECH('nCovariates', 0);
            theta = mdl.samplePrior(1);
            theta(3) = abs(theta(3));                 % sigma_eta > 0
            [y, h] = mdl.simulate(theta, 120);
            omega  = mdl.omegaPath(theta, h, y);
            testCase.verifySize(omega, [120, 1]);
            s = mdl.unpack(theta);
            testCase.verifyEqual(omega(1), s.beta_0, 'RelTol', 1e-12);
            testCase.verifyTrue(all(isfinite(omega)));
        end

        function withCovariates(testCase)
            rng(6, 'threefry');
            K   = 2; T = 100;
            Z   = randn(T, K);
            mdl = models.SVLTRECH('nCovariates', K, 'covariates', Z);
            theta = mdl.samplePrior(1); theta(3) = abs(theta(3));
            [y, h] = mdl.simulate(theta, T);
            omega  = mdl.omegaPath(theta, h, y);
            testCase.verifySize(omega, [T, 1]);
            testCase.verifyTrue(all(isfinite(omega)));
        end
    end
end
