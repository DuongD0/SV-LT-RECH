classdef tRnnCellsGru < matlab.unittest.TestCase
% tRnnCellsGru  Numerical correctness of the +cells/gru.m forward pass.

    methods (Static)
        function w = zeroWeights(d)
            z = zeros(1, d);
            w = struct('W_z', z, 'u_z', 0, 'b_z', 0, ...
                       'W_r', z, 'u_r', 0, 'b_r', 0, ...
                       'W_h', z, 'u_h', 0, 'b_h', 0);
        end
    end

    methods (Test)

        function zeroStateZeroWeightsGivesZero(testCase)
            % z=r=sigmoid(0)=0.5, candidate=tanh(0)=0,
            % sNew = (1-0.5)*0 + 0.5*0 = 0; ctx empty.
            w = tRnnCellsGru.zeroWeights(3);
            [s, ctx] = cells.gru(zeros(5, 3), zeros(5, 1), w);

            testCase.verifyEqual(s, zeros(5, 1), 'AbsTol', 1e-12);
            testCase.verifyEqual(ctx, struct());
        end

        function updateGateClosedKeepsPrev(testCase)
            % z≈0 (b_z=-100) => sNew = 1*sPrev + 0 = sPrev.
            w = tRnnCellsGru.zeroWeights(2);
            w.b_z = -100;
            sPrev = [0.4; -1.2; 3.0];
            s = cells.gru(zeros(3, 2), sPrev, w);

            testCase.verifyEqual(s, sPrev, 'AbsTol', 1e-9);
        end

        function updateGateOpenTakesCandidate(testCase)
            % z≈1 (b_z=100), u_h=0 => sNew = tanh(W_h x + b_h).
            w = tRnnCellsGru.zeroWeights(1);
            w.b_z = 100;
            w.W_h = 1;
            x = [0.2; -0.5; 1.1];
            s = cells.gru(x, [7; 7; 7], w);  % sPrev irrelevant (z≈1, u_h=0)

            testCase.verifyEqual(s, tanh(x), 'AbsTol', 1e-9);
        end

        function resetGateClosedZeroesRecurrentInCandidate(testCase)
            % r≈0 (b_r=-100), z≈1 (b_z=100), large u_h. The reset gate must
            % kill the recurrent term so candidate = tanh(W_h x + b_h).
            w = tRnnCellsGru.zeroWeights(1);
            w.b_z = 100; w.b_r = -100;
            w.W_h = 1; w.u_h = 5;
            x = [0.3; -0.8];
            s = cells.gru(x, [10; -10], w);  % big sPrev, but reset gate ≈ 0

            testCase.verifyEqual(s, tanh(x), 'AbsTol', 1e-9);
        end

        function vectorisedSameAsLoop(testCase)
            rng(11, 'threefry');
            d = 4; n = 40;
            mk = @() randn(1, d) * 0.1;
            w = struct('W_z', mk(), 'u_z', 0.1, 'b_z', 0.05, ...
                       'W_r', mk(), 'u_r', -0.2, 'b_r', 0.0, ...
                       'W_h', mk(), 'u_h', 0.3, 'b_h', 0.1);
            x = randn(n, d);
            sPrev = randn(n, 1);

            sVec = cells.gru(x, sPrev, w);

            sig = @(z) 0.5 * (1 + tanh(z / 2));
            sLoop = zeros(n, 1);
            for k = 1:n
                zg = sig(x(k,:) * w.W_z(:) + w.u_z * sPrev(k) + w.b_z);
                rg = sig(x(k,:) * w.W_r(:) + w.u_r * sPrev(k) + w.b_r);
                g  = tanh(x(k,:) * w.W_h(:) + w.u_h * (rg * sPrev(k)) + w.b_h);
                sLoop(k) = (1 - zg) * sPrev(k) + zg * g;
            end
            testCase.verifyEqual(sVec, sLoop, 'AbsTol', 1e-12);
        end

        function sizeMismatchErrors(testCase)
            w = tRnnCellsGru.zeroWeights(2);
            testCase.verifyError(@() cells.gru(randn(5, 2), randn(3, 1), w), ...
                'gru:sizeMismatch');
        end

    end
end
