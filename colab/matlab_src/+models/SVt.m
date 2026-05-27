classdef SVt < models.Model
% SVt  Stochastic volatility with Student-t return innovations (§7.5b).
%
%   Observation:  r_t = exp(h_t/2) * eps_t,
%                 eps_t = sqrt((nu-2)/nu) * T_nu,    T_nu ~ Student-t_nu
%
%     The scaling sqrt((nu-2)/nu) gives Var(eps_t) = 1 so that
%     Var(r_t | h_t) = exp(h_t) and the volatility interpretation of h_t
%     is identical to plain SV. nu > 2 is required (enforced in the prior).
%
%   Latent state: h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,
%                 eta_t ~ N(0, 1),  |phi| < 1,  sigma_eta > 0
%
%   theta = [mu, phi, sigma_eta, nu]
%
%   Priors and DGP follow PROPOSED_METHODOLOGY.md §7.5b. See
%   priors.logPriorSVt for the prior log density.

    methods

        function names = paramNames(~)
            names = {'mu', 'phi', 'sigma_eta', 'nu'};
        end

        %% ---- Priors -----------------------------------------------

        function lp = logPrior(~, theta)
            lp = priors.logPriorSVt(theta);
        end

        function theta = samplePrior(~, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));
            % Truncated Exponential(rate=0.1) shifted to nu > 2.
            nu        = 2 + exprnd(10, n, 1);
            theta     = [mu, phi, sigma_eta, nu];
        end

        %% ---- Latent state ----------------------------------------

        function h0 = initLatent(obj, theta, n)
            s = obj.unpack(theta);
            stationarySd = s.sigma_eta / sqrt(1 - s.phi^2);
            h0 = s.mu + stationarySd * randn(n, 1);
        end

        function [hNew, auxNew] = transitionSample(obj, hOld, theta, ~, n, aux)
            s = obj.unpack(theta);
            hNew   = s.mu + s.phi * (hOld - s.mu) + s.sigma_eta * randn(n, 1);
            auxNew = aux;
        end

        %% ---- Observation -----------------------------------------

        function logp = observationLogLik(obj, yT, hT, theta)
            % Scaled Student-t density: location 0, scale exp(h/2)*sqrt((nu-2)/nu),
            % df nu. After algebra (see derivation in docs/design_decisions.md):
            %   logp = gammaln((nu+1)/2) - gammaln(nu/2)
            %        - 0.5*log(pi*(nu-2)) - h/2
            %        - ((nu+1)/2) * log1p( y^2 / (exp(h) * (nu-2)) )
            s  = obj.unpack(theta);
            nu = s.nu;
            scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
            logp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
                 - 0.5 * log(pi * (nu - 2)) ...
                 - 0.5 * hT ...
                 - ((nu + 1) / 2) * log1p(scaledSq);
        end

        %% ---- DGP --------------------------------------------------

        function [y, h] = simulate(obj, theta, T, varargin)
            s        = obj.unpack(theta);
            sigmaEps = sqrt((s.nu - 2) / s.nu);
            h = zeros(T, 1);
            y = zeros(T, 1);
            h(1) = s.mu + s.sigma_eta / sqrt(1 - s.phi^2) * randn();
            y(1) = exp(h(1) / 2) * sigmaEps * trnd(s.nu);
            for t = 2:T
                h(t) = s.mu + s.phi * (h(t-1) - s.mu) + s.sigma_eta * randn();
                y(t) = exp(h(t) / 2) * sigmaEps * trnd(s.nu);
            end
        end

        %% ---- Packing ---------------------------------------------

        function s = unpack(~, theta)
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
            s.nu        = theta(4);
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.nu];
        end
    end
end
