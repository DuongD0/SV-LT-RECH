classdef tStage4Smoke < matlab.unittest.TestCase
% tStage4Smoke  Network-free end-to-end Stage-4 pipeline on a synthetic fixture.

    methods (TestMethodSetup)
        function requireEcon(testCase)
            testCase.assumeTrue(~isempty(ver('econ')), ...
                'Econometrics Toolbox not available; skipping Stage-4 smoke.');
        end
    end

    methods (Test)
        function pipelineProducesScoresForAllModels(testCase)
            rng(99, 'threefry');

            % Synthetic VN-like series from the flagship DGP with 2 covariates.
            K        = 2;
            T        = 280;
            splitIdx = 210;
            thetaTrue = [-0.5, 0.95, 0.25, -0.4, 8, 0.05, 0.20, ...
                         0.05, -0.05, 0.05, 0.10, -0.10, 0.10, 0.0];  % 12 + 2
            Z   = randn(T, K);
            gen = models.SVLTRECH('nCovariates', K, 'leverage', 'cholesky');
            gen.setCovariates(Z);
            [y, ~] = gen.simulate(thetaTrue, T);

            cfg = struct();
            cfg.model    = struct('nCovariates', K, 'leverage', 'cholesky');
            cfg.smc      = struct('N', 250, 'M', 40, 'nSweeps', 2, ...
                                  'proposalScale', 0.5, 'essThreshold', 0.5, ...
                                  'targetEss', 0.5, 'verbose', false);
            cfg.forecast = struct('J', 20);
            cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 200);
            cfg.particleFilter = struct('clipLogWeight', -50);
            cfg.baseSeed = 20260516;

            out = experiments.runStage4(y, Z, splitIdx, cfg);

            expected = {'GARCH-t','GJR-t','SVLT','SVLTRECH'};
            testCase.verifyEqual(out.modelNames, expected);

            S = out.scores;     % struct array, one per model
            testCase.verifyEqual(numel(S), 4);
            for m = 1:numel(S)
                testCase.verifyTrue(isfinite(S(m).pps));
                testCase.verifyTrue(isfinite(S(m).qlike));
                testCase.verifyTrue(isfinite(S(m).mse));
                testCase.verifyTrue(isfinite(S(m).mae));
                testCase.verifyTrue(isfinite(S(m).r2log));
                testCase.verifyTrue(isfinite(S(m).qs1));
                testCase.verifyTrue(isfinite(S(m).qs5));
            end

            % MCS keeps at least one model; DM table has 3 pairwise rows.
            testCase.verifyGreaterThanOrEqual(sum(out.mcs.inSet), 1);
            testCase.verifyEqual(numel(out.dmVsProposed), 3);
        end
    end
end
