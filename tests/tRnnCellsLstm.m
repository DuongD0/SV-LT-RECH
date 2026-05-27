classdef tRnnCellsLstm < matlab.unittest.TestCase
% tRnnCellsLstm  Numerical correctness of the +cells/lstm.m forward pass.

    methods (Static)
        function w = zeroWeights(d)
            z = zeros(1, d);
            w = struct('W_f', z, 'u_f', 0, 'b_f', 0, ...
                       'W_i', z, 'u_i', 0, 'b_i', 0, ...
                       'W_o', z, 'u_o', 0, 'b_o', 0, ...
                       'W_c', z, 'u_c', 0, 'b_c', 0);
        end
    end

    methods (Test)

        function zeroStateZeroWeightsGivesZero(testCase)
            % All gates = sigmoid(0) = 0.5, candidate = tanh(0) = 0,
            % so c = 0.5*0 + 0.5*0 = 0 and s = 0.5*tanh(0) = 0.
            w = tRnnCellsLstm.zeroWeights(3);
            x = zeros(5, 3);
            [s, ctx] = cells.lstm(x, zeros(5, 1), zeros(5, 1), w);

            testCase.verifyEqual(s, zeros(5, 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(ctx.c, zeros(5, 1), 'AbsTol', 1e-12);
        end

        function forgetGateOpenRetainsCellState(testCase)
            % f≈1 (b_f=100), i≈0 (b_i=-100), o≈1 (b_o=100), all W,u=0.
            % => c_t = c_{t-1}, s_t = tanh(c_{t-1}).
            w = tRnnCellsLstm.zeroWeights(2);
            w.b_f = 100; w.b_i = -100; w.b_o = 100;
            cPrev = [0.5; 1.0; 2.0];
            x = zeros(3, 2);
            [s, ctx] = cells.lstm(x, zeros(3, 1), cPrev, w);

            testCase.verifyEqual(ctx.c, cPrev, 'AbsTol', 1e-9);
            testCase.verifyEqual(s, tanh(cPrev), 'AbsTol', 1e-9);
        end

        function inputGateWritesCandidate(testCase)
            % f≈0, i≈1, o≈1; candidate driven by W_c·x with x scalar input.
            % => c_t = tanh(W_c x), s_t = tanh(c_t).
            w = tRnnCellsLstm.zeroWeights(1);
            w.b_f = -100; w.b_i = 100; w.b_o = 100;
            w.W_c = 1;
            x = [0.3; -0.7; 1.5];
            [s, ctx] = cells.lstm(x, zeros(3, 1), [9; 9; 9], w);  % cPrev irrelevant (f≈0)

            cExpected = tanh(x);
            testCase.verifyEqual(ctx.c, cExpected, 'AbsTol', 1e-9);
            testCase.verifyEqual(s, tanh(cExpected), 'AbsTol', 1e-9);
        end

        function vectorisedSameAsLoop(testCase)
            rng(7, 'threefry');
            d = 4; n = 40;
            mk = @() randn(1, d) * 0.1;
            w = struct('W_f', mk(), 'u_f', 0.1, 'b_f', 0.05, ...
                       'W_i', mk(), 'u_i', -0.2, 'b_i', 0.0, ...
                       'W_o', mk(), 'u_o', 0.15, 'b_o', -0.05, ...
                       'W_c', mk(), 'u_c', 0.3, 'b_c', 0.1);
            x = randn(n, d);
            sPrev = randn(n, 1);
            cPrev = randn(n, 1);

            [sVec, ctxVec] = cells.lstm(x, sPrev, cPrev, w);

            sig = @(z) 0.5 * (1 + tanh(z / 2));
            sLoop = zeros(n, 1); cLoop = zeros(n, 1);
            for k = 1:n
                f = sig(x(k,:) * w.W_f(:) + w.u_f * sPrev(k) + w.b_f);
                i = sig(x(k,:) * w.W_i(:) + w.u_i * sPrev(k) + w.b_i);
                o = sig(x(k,:) * w.W_o(:) + w.u_o * sPrev(k) + w.b_o);
                g = tanh(x(k,:) * w.W_c(:) + w.u_c * sPrev(k) + w.b_c);
                cLoop(k) = f * cPrev(k) + i * g;
                sLoop(k) = o * tanh(cLoop(k));
            end
            testCase.verifyEqual(sVec, sLoop, 'AbsTol', 1e-12);
            testCase.verifyEqual(ctxVec.c, cLoop, 'AbsTol', 1e-12);
        end

        function sPrevSizeMismatchErrors(testCase)
            w = tRnnCellsLstm.zeroWeights(2);
            testCase.verifyError(@() cells.lstm(randn(5, 2), randn(3, 1), randn(5, 1), w), ...
                'lstm:sizeMismatch');
        end

        function cPrevSizeMismatchErrors(testCase)
            w = tRnnCellsLstm.zeroWeights(2);
            testCase.verifyError(@() cells.lstm(randn(5, 2), randn(5, 1), randn(3, 1), w), ...
                'lstm:sizeMismatch');
        end

    end
end
