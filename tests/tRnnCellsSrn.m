classdef tRnnCellsSrn < matlab.unittest.TestCase
% tRnnCellsSrn  Numerical correctness of the +cells/srn.m forward pass.

    methods (Test)

        function zeroInputWithZeroBiasGivesZero(testCase)
            weights = struct('v', [0, 0, 0], 'w_h', 0, 'b', 0);
            x       = zeros(5, 3);
            sPrev   = zeros(5, 1);
            [s, ctx] = cells.srn(x, sPrev, weights);

            testCase.verifyEqual(s, zeros(5, 1));
            testCase.verifyEqual(ctx, struct());
        end

        function reluClampsNegativeToZero(testCase)
            weights = struct('v', [1, 1], 'w_h', 0, 'b', -100);
            x       = [1, 2; 3, 4; -5, -10];
            sPrev   = zeros(3, 1);
            s       = cells.srn(x, sPrev, weights);

            testCase.verifyEqual(s, zeros(3, 1));
        end

        function reluPassesPositive(testCase)
            weights = struct('v', [1, 2], 'w_h', 0, 'b', 0);
            x       = [1, 1; 0, 0; -1, 2];
            sPrev   = zeros(3, 1);
            s       = cells.srn(x, sPrev, weights);

            % Pre-activations: 1*1+2*1=3; 0; -1+4=3.
            testCase.verifyEqual(s, [3; 0; 3]);
        end

        function recurrenceAddsWeightedPrev(testCase)
            weights = struct('v', 0, 'w_h', 0.5, 'b', 0);
            x       = zeros(4, 1);
            sPrev   = [2; 4; -3; 0];
            s       = cells.srn(x, sPrev, weights);

            % ReLU(0.5 * sPrev) = [1; 2; 0; 0]
            testCase.verifyEqual(s, [1; 2; 0; 0]);
        end

        function singleParticleVectorisedSameAsLoop(testCase)
            rng(99, 'threefry');
            d       = 4;
            n       = 50;
            weights = struct('v', randn(1, d) * 0.1, ...
                             'w_h', 0.2, ...
                             'b', 0.05);
            x       = randn(n, d);
            sPrev   = abs(randn(n, 1));

            sVec  = cells.srn(x, sPrev, weights);
            sLoop = zeros(n, 1);
            for i = 1:n
                sLoop(i) = max(x(i, :) * weights.v(:) ...
                               + weights.w_h * sPrev(i) + weights.b, 0);
            end
            testCase.verifyEqual(sVec, sLoop, 'AbsTol', 1e-12);
        end

        function sizeMismatchErrors(testCase)
            weights = struct('v', [1, 1], 'w_h', 0, 'b', 0);
            x       = randn(5, 2);
            sPrev   = randn(3, 1);
            testCase.verifyError(@() cells.srn(x, sPrev, weights), ...
                'srn:sizeMismatch');
        end

    end
end
