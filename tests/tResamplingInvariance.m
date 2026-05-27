classdef tResamplingInvariance < matlab.unittest.TestCase
% tResamplingInvariance  Resampling preserves the weighted empirical mean.
%
%   On average, mean(x(idx)) over many resampling draws should converge
%   to the weighted mean sum_i w_i * x_i of the input.

    methods (Test)

        function systematicPreservesMean(testCase)
            seed = 20260516;
            utils.reproducibility(seed);

            N       = 200;
            x       = randn(N, 1);
            logWraw = randn(N, 1) * 2;
            wNorm   = exp(logWraw - utils.logsumexp(logWraw));
            wMean   = sum(wNorm .* x);

            nDraws = 2000;
            means  = zeros(nDraws, 1);
            for k = 1:nDraws
                idx      = inference.pf.resampleSystematic(logWraw);
                means(k) = mean(x(idx));
            end

            testCase.verifyEqual(mean(means), wMean, 'AbsTol', 0.05);
        end

        function stratifiedPreservesMean(testCase)
            seed = 20260517;
            utils.reproducibility(seed);

            N       = 200;
            x       = randn(N, 1);
            logWraw = randn(N, 1) * 2;
            wNorm   = exp(logWraw - utils.logsumexp(logWraw));
            wMean   = sum(wNorm .* x);

            nDraws = 2000;
            means  = zeros(nDraws, 1);
            for k = 1:nDraws
                idx      = inference.pf.resampleStratified(logWraw);
                means(k) = mean(x(idx));
            end

            testCase.verifyEqual(mean(means), wMean, 'AbsTol', 0.05);
        end

        function indicesInRange(testCase)
            seed = 20260518;
            utils.reproducibility(seed);
            N    = 200;
            logW = randn(N, 1) * 2;
            idx  = inference.pf.resampleSystematic(logW);
            testCase.verifyGreaterThanOrEqual(min(idx), 1);
            testCase.verifyLessThanOrEqual(max(idx), N);
            testCase.verifyEqual(numel(idx), N);
        end
    end
end
