classdef tEssGradient < matlab.unittest.TestCase
% tEssGradient  ESS should be monotone non-increasing in temperature.
%
%   As we anneal further (larger a), particle weights become more
%   informative and ESS drops. Catches reweighting bugs that produce
%   non-monotone or negative ESS.

    methods (Test)

        function essMonotoneInTemperature(testCase)
            seed = 20260516;
            utils.reproducibility(seed);

            N    = 1000;
            logW = zeros(N, 1) - log(N);
            llh  = randn(N, 1) * 5;

            grid    = linspace(0, 1, 25)';
            essVals = zeros(numel(grid), 1);
            for i = 1:numel(grid)
                essVals(i) = inference.smc.ess(logW + grid(i) * llh);
            end

            diffs = diff(essVals);
            testCase.verifyLessThanOrEqual(diffs, 1e-6 * N);
        end

        function essAtZeroIsN(testCase)
            N    = 1000;
            logW = zeros(N, 1) - log(N);
            testCase.verifyEqual(inference.smc.ess(logW), N, 'AbsTol', 1e-6);
        end

        function essBoundedBetweenOneAndN(testCase)
            N = 500;
            for trial = 1:10
                logW = randn(N, 1) * 3;
                e    = inference.smc.ess(logW);
                testCase.verifyGreaterThanOrEqual(e, 1 - 1e-9);
                testCase.verifyLessThanOrEqual(e, N + 1e-9);
            end
        end
    end
end
