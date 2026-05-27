classdef tLeverageCoupling < matlab.unittest.TestCase
% tLeverageCoupling  Statistical properties of both leverage coupling
%                    strategies (Cholesky + OCSN-mixture).

    methods (Test)

        function choleskyMarginalIsStandardNormal(testCase)
            rng(1, 'threefry');
            n   = 50000;
            eps = randn(n, 1);
            rho = 0.5;
            eta = utils.leverageCholesky(rho, eps, n);

            testCase.verifyEqual(mean(eta), 0, 'AbsTol', 0.02);
            testCase.verifyEqual(var(eta),  1, 'RelTol', 0.05);
        end

        function choleskyCorrelationMatchesRho(testCase)
            rng(2, 'threefry');
            n      = 200000;
            rhoVec = [-0.7, -0.3, 0, 0.3, 0.7];

            for rho = rhoVec
                eps    = randn(n, 1);
                eta    = utils.leverageCholesky(rho, eps, n);
                rhoHat = corr(eps, eta);
                testCase.verifyEqual(rhoHat, rho, 'AbsTol', 0.01, ...
                    sprintf('Cholesky rho=%.2f: sample corr=%.4f', rho, rhoHat));
            end
        end

        function ocsnMarginalIsApproxStandardNormal(testCase)
            rng(3, 'threefry');
            n   = 50000;
            eps = randn(n, 1);
            rho = 0.5;
            eta = utils.leverageOcsn(rho, eps, n);

            testCase.verifyEqual(mean(eta), 0, 'AbsTol', 0.03);
            testCase.verifyEqual(var(eta),  1, 'RelTol', 0.05);
        end

        function ocsnCorrelationIsSignAndScaleConsistent(testCase)
            rng(4, 'threefry');
            n      = 200000;
            rhoVec = [-0.7, -0.3, 0.3, 0.7];

            for rho = rhoVec
                eps    = randn(n, 1);
                eta    = utils.leverageOcsn(rho, eps, n);
                rhoHat = corr(eps, eta);

                testCase.verifyEqual(sign(rhoHat), sign(rho), ...
                    sprintf('OCSN rho=%.2f: sign(corr)=%d', rho, sign(rhoHat)));
                testCase.verifyGreaterThan(abs(rhoHat), 0.3 * abs(rho), ...
                    sprintf('OCSN rho=%.2f: |corr|=%.3f too dampened', rho, abs(rhoHat)));
                testCase.verifyLessThan(abs(rhoHat), abs(rho) + 0.05, ...
                    sprintf('OCSN rho=%.2f: |corr|=%.3f exceeds rho', rho, abs(rhoHat)));
            end

            eps = randn(n, 1);
            eta = utils.leverageOcsn(0, eps, n);
            testCase.verifyEqual(corr(eps, eta), 0, 'AbsTol', 0.01);
        end

        function inputValidation(testCase)
            testCase.verifyError(@() utils.leverageCholesky(1.0, randn(5,1), 5), ...
                'MATLAB:validators:mustBeLessThan');
            testCase.verifyError(@() utils.leverageOcsn(-1.0, randn(5,1), 5), ...
                'MATLAB:validators:mustBeGreaterThan');
        end

        function sizeContract(testCase)
            eps   = randn(37, 1);
            etaCh = utils.leverageCholesky(0.3, eps, 37);
            etaOc = utils.leverageOcsn(0.3, eps, 37);

            testCase.verifySize(etaCh, [37, 1]);
            testCase.verifySize(etaOc, [37, 1]);
        end

    end
end
