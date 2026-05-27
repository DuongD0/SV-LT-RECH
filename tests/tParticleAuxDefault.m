classdef tParticleAuxDefault < matlab.unittest.TestCase
% tParticleAuxDefault  Regression test for the Model.m aux-hook extension.
%
%   The Phase 3 interface change added `initParticleAux` and
%   `updateParticleAux` hooks to models.Model and threaded `aux` through
%   the bootstrap PF. Plain SV does not use them. This test asserts:
%
%     1. Default `initParticleAux` returns a usable struct.
%     2. Default `updateParticleAux` is a pass-through.
%     3. The bootstrap PF still produces a finite log-likelihood on a
%        simulated SV path.
%     4. The PF is deterministic under a fixed RNG seed.

    methods (Test)

        function defaultAuxIsStruct(testCase)
            mdl   = models.SV();
            theta = [0, 0.95, 0.4];
            aux   = mdl.initParticleAux(theta, 100);
            testCase.verifyClass(aux, 'struct');
        end

        function defaultUpdateIsPassThrough(testCase)
            mdl   = models.SV();
            theta = [0, 0.95, 0.4];
            aux0  = mdl.initParticleAux(theta, 50);
            h     = randn(50, 1);
            [aux1, ctx] = mdl.updateParticleAux(aux0, h, 0.5, theta, 7);

            testCase.verifyEqual(aux1, aux0);
            testCase.verifyClass(ctx, 'struct');
        end

        function bootstrapPfStillReturnsFiniteLogLik(testCase)
            rng(7, 'threefry');
            mdl   = models.SV();
            theta = [0, 0.95, 0.4];
            [y, ~] = mdl.simulate(theta, 200);

            logLik = inference.pf.bootstrap(mdl, y, theta, struct('M', 200));

            testCase.verifyTrue(isfinite(logLik));
            testCase.verifyLessThan(logLik, 0);
        end

        function pfDeterministicUnderFixedSeed(testCase)
            mdl   = models.SV();
            theta = [0, 0.95, 0.4];

            rng(42, 'threefry');
            [y, ~] = mdl.simulate(theta, 100);

            rng(123, 'threefry');
            l1 = inference.pf.bootstrap(mdl, y, theta, struct('M', 100));

            rng(123, 'threefry');
            l2 = inference.pf.bootstrap(mdl, y, theta, struct('M', 100));

            testCase.verifyEqual(l1, l2, 'RelTol', 0);
        end

    end
end
