classdef tBootstrapIncrements < matlab.unittest.TestCase
% tBootstrapIncrements  PF per-step predictive density + variance outputs.

    methods (Test)

        function incrementsSumToScalarLogLik(testCase)
            rng(2026, 'threefry');
            mdl   = models.SV();
            theta = [-0.2, 0.95, 0.30];                 % [mu, phi, sigma_eta]
            [y, ~] = mdl.simulate(theta, 300);

            rng(7, 'threefry');
            logLik = inference.pf.bootstrap(mdl, y, theta, struct('M', 400));

            rng(7, 'threefry');                          % same PF randomness
            [logLik2, ~, lpd, vf] = inference.pf.bootstrap( ...
                mdl, y, theta, struct('M', 400, 'returnIncrements', true));

            testCase.verifyEqual(logLik2, logLik, 'RelTol', 1e-10);
            testCase.verifyEqual(numel(lpd), numel(y));
            testCase.verifyEqual(sum(lpd), logLik2, 'RelTol', 1e-10);
            testCase.verifySize(vf, [numel(y), 1]);
            testCase.verifyTrue(all(vf > 0));
            testCase.verifyTrue(all(isfinite(lpd)));
        end

        function defaultCallUnchanged(testCase)
            rng(3, 'threefry');
            mdl   = models.SV();
            theta = [0.0, 0.9, 0.4];
            [y, ~] = mdl.simulate(theta, 200);
            % Old 2-output contract still works.
            [logLik, hF] = inference.pf.bootstrap(mdl, y, theta, ...
                struct('M', 200, 'returnPath', true));
            testCase.verifyTrue(isscalar(logLik) && isfinite(logLik));
            testCase.verifySize(hF, [numel(y), 1]);
        end
    end
end
