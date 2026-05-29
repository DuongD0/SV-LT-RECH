classdef tBdsTest < matlab.unittest.TestCase
% tBdsTest  BDS statistic: does not over-reject iid, rejects dependence.

    methods (Test)

        function iidNotRejected(testCase)
            rng(101, 'threefry');
            x = randn(1500, 1);                 % iid Gaussian -> H0 true
            out = data.preprocess.bdsTest(x, 2, 0.5);
            testCase.verifyTrue(isfinite(out.stat));
            testCase.verifyGreaterThanOrEqual(out.pValue, 0);
            testCase.verifyLessThanOrEqual(out.pValue, 1);
            testCase.verifyLessThan(abs(out.stat), 3.0);   % ~no rejection
        end

        function archDependenceRejected(testCase)
            rng(202, 'threefry');
            T = 1500; x = zeros(T, 1); h = 1;
            for t = 2:T
                h = 0.2 + 0.7 * x(t-1)^2 + 0.1 * h;     % volatility clustering
                x(t) = sqrt(h) * randn();
            end
            out = data.preprocess.bdsTest(x, 2, 0.5);
            testCase.verifyGreaterThan(abs(out.stat), 5.0);  % strong rejection
            testCase.verifyLessThan(out.pValue, 1e-3);
        end

        function embeddingDimDefaults(testCase)
            rng(7, 'threefry');
            out = data.preprocess.bdsTest(randn(400, 1));    % defaults m=2, eps=0.5
            testCase.verifyTrue(isfinite(out.stat));
        end
    end
end
