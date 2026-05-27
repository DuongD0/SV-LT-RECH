classdef tSimulatorRoundTrip < matlab.unittest.TestCase
% tSimulatorRoundTrip  Phase 3 CI gate -- code-path smoke for the full
%                      Stage 1 pipeline.
%
%   Runs experiments.runStage1 on the very small stage1_smoke.yaml. The
%   purpose is to confirm the WHOLE PIPELINE -- SVLTRECH simulator,
%   leverage coupling, RNN cell, bootstrap PF, likelihood-annealing SMC,
%   diagnostics, output aggregation -- runs end-to-end without errors and
%   produces sensible structured output.
%
%   This is NOT a convergence gate. At smoke scale (N=200, M=40, T=200,
%   2 replicates x 2 seeds) the 12-parameter SVLTRECH posterior is
%   nowhere near converged -- R-hat will be large, coverage will be
%   noisy, MCSE of log Z will be high. Those are FULL-STUDY gates in
%   phase log §4.5, exercised manually via scripts/run_stage1_simulation.m.

    methods (Test)

        function smokeStudyRunsEndToEnd(testCase)
            configPath = fullfile('config', 'experiments', 'stage1_smoke.yaml');

            opts.writeOutputs = false;
            opts.outDir       = tempname;
            out               = experiments.runStage1(string(configPath), opts);

            %% Structural: every expected field is present and well-shaped.
            testCase.verifyTrue(isfield(out, 'summary'));
            testCase.verifyTrue(isfield(out, 'pnames'));
            testCase.verifyTrue(isfield(out, 'thetaTrue'));
            testCase.verifyTrue(isfield(out, 'rhatPerReplicate'));
            testCase.verifyTrue(isfield(out, 'postMeanPerReplicate'));
            testCase.verifyTrue(isfield(out, 'postStdPerReplicate'));
            testCase.verifyTrue(isfield(out, 'fracFlipped'));

            K = numel(out.thetaTrue);
            R = size(out.rhatPerReplicate, 1);
            testCase.verifySize(out.summary.coverage95, [1, K]);
            testCase.verifySize(out.postMeanPerReplicate, [R, K]);
            testCase.verifySize(out.postStdPerReplicate,  [R, K]);

            %% Numerical sanity: nothing exploded.
            testCase.verifyTrue(all(isfinite(out.postMeanPerReplicate(:))), ...
                'postMeanPerReplicate has non-finite entries');
            testCase.verifyTrue(all(out.postStdPerReplicate(:) > 0), ...
                'postStdPerReplicate has non-positive entries');
            testCase.verifyTrue(all(isfinite(out.rhatPerReplicate(:))), ...
                'rhatPerReplicate has non-finite entries');
            testCase.verifyTrue(all(isfinite(out.summary.meanLogZ)), ...
                'meanLogZ has non-finite entries');

            %% Hard prior constraints: beta_1 >= 0 always.
            testCase.verifyEqual(out.summary.maxFracFlipped, 0, ...
                sprintf('maxFracFlipped = %.3f (prior enforces beta_1 >= 0)', ...
                        out.summary.maxFracFlipped));

            %% R-hat is finite and reasonable. We do NOT require <1.2 at
            %% smoke scale; the full-study gate (manual) demands <=1.10.
            testCase.verifyLessThan(out.summary.maxRhat, 10, ...
                sprintf('maxRhat = %.3f (smoke ceiling, not the §4.5 gate)', ...
                        out.summary.maxRhat));
        end

    end
end
