classdef tPriorsDensity < matlab.unittest.TestCase
% tPriorsDensity  Unit tests for prior log-density implementations.
%
%   Cheapest test that catches sign errors and bad Jacobians.

    methods (Test)

        function svPriorMatchesHand(testCase)
            % Hand calc of logPriorSV at theta = [0, 0.9, 0.5].
            mu = 0; phi = 0.9; sig = 0.5;
            sd = 10;
            lp_mu_hand = -0.5*(mu/sd)^2 - log(sd) - 0.5*log(2*pi);
            ph  = (1+phi)/2;
            lp_phi_hand = 19*log(ph) + 0.5*log(1-ph) - betaln(20, 1.5) - log(2);
            lp_sig_hand = log(2) - log(pi) - log(1 + sig^2);
            expected = lp_mu_hand + lp_phi_hand + lp_sig_hand;

            actual = priors.logPriorSV([mu, phi, sig]);
            testCase.verifyEqual(actual, expected, 'AbsTol', 1e-10);
        end

        function outOfSupportReturnsNegInf(testCase)
            testCase.verifyEqual(priors.logPriorSV([0,  1.5, 1.0]), -Inf);
            testCase.verifyEqual(priors.logPriorSV([0,  0.9,-1.0]), -Inf);
            testCase.verifyEqual(priors.logPriorSV([0, -1.0, 1.0]), -Inf);
        end

        function priorSamplesAreInSupport(testCase)
            mdl   = models.SV();
            theta = mdl.samplePrior(500);
            for i = 1:size(theta, 1)
                testCase.verifyTrue(isfinite(mdl.logPrior(theta(i, :))));
            end
        end
    end
end
