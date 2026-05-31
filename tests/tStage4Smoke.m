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
            cfg.smc      = struct('N', 200, 'M', 30, 'nSweeps', 2, ...
                                  'proposalScale', 0.5, 'essThreshold', 0.5, ...
                                  'targetEss', 0.5, 'verbose', false);
            cfg.forecast = struct('J', 15);
            cfg.eval     = struct('alphaQS', [0.01, 0.05], 'mcsB', 200);
            cfg.particleFilter = struct('clipLogWeight', -50);
            cfg.garchRech = struct('restarts', 1, 'maxEval', 800);   % fast smoke
            cfg.baseSeed = 20260516;

            out = experiments.runStage4(y, Z, splitIdx, cfg);

            expected = {'GARCH-t','GJR-t', ...
                        'GARCH-RECH-SRN','GARCH-RECH-LSTM','GARCH-RECH-GRU', ...
                        'SV','SVM','SVLT', ...
                        'SVLTRECH-SRN','SVLTRECH-LSTM','SVLTRECH-GRU'};
            testCase.verifyEqual(out.modelNames, expected);

            S = out.scores;
            testCase.verifyEqual(numel(S), 11);
            for m = 1:numel(S)
                testCase.verifyTrue(isfinite(S(m).pps));
                testCase.verifyTrue(isfinite(S(m).qlike));
                testCase.verifyTrue(isfinite(S(m).mse));
                testCase.verifyTrue(isfinite(S(m).r2log));
            end

            testCase.verifyGreaterThanOrEqual(sum(out.mcs.inSet), 1);
            testCase.verifyEqual(numel(out.dmVsProposed), 10);  % 11 - 1 vs proposed

            % Bundle contract
            b = out.bundle;
            testCase.verifyEqual(numel(b.models), 11);
            testCase.verifyTrue(isfield(b.data, 'meanEquation'));
            testCase.verifyTrue(isfield(b, 'descriptives'));
            testCase.verifyTrue(isfield(b.descriptives, 'bds'));
            % SV models carry posterior particles + log-ML; GARCH do not.
            srn = b.models(strcmp({b.models.name}, 'SVLTRECH-SRN'));
            testCase.verifyTrue(isfinite(srn.logMarginalLik));
            testCase.verifySize(srn.theta, [cfg.smc.N, 12 + K]);
            testCase.verifyEqual(numel(srn.stdResid), splitIdx);
            testCase.verifyEqual(numel(srn.omegaPath), splitIdx);
        end
    end
end
