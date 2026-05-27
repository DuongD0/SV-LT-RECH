classdef SV < models.Model
% SV  Plain stochastic volatility (Kim, Shephard, Chib 1998).
%
%   Observation:  r_t  = exp(h_t / 2) * eps_t,   eps_t ~ N(0, 1)
%   Latent state: h_t  = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,
%                 eta_t ~ N(0, 1),    |phi| < 1,   sigma_eta > 0
%
%   Initial state: h_1 ~ N(mu, sigma_eta^2 / (1 - phi^2))   (stationary)
%
%   theta = [mu, phi, sigma_eta]
%
%   This is the reference baseline (§7.5a) and the cross-engine calibration
%   target against R `stochvol::svsample`.

    methods

        function names = paramNames(~)
            names = {'mu', 'phi', 'sigma_eta'};
        end

        %% ---- Priors -----------------------------------------------

        function lp = logPrior(obj, theta)
            lp = priors.logPriorSV(theta);
        end

        function theta = samplePrior(obj, n)
            % Rejection-style: sample each parameter from its marginal prior.
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));   % half-Cauchy
            theta     = [mu, phi, sigma_eta];
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

        function logp = observationLogLik(obj, yT, hT, ~)
            % log N(y_t; 0, exp(h_t))
            logp = -0.5 * log(2 * pi) - 0.5 * hT - 0.5 * yT.^2 .* exp(-hT);
        end

        %% ---- DGP --------------------------------------------------

        function [y, h] = simulate(obj, theta, T, varargin)
            s = obj.unpack(theta);
            h = zeros(T, 1);
            y = zeros(T, 1);
            h(1) = s.mu + s.sigma_eta / sqrt(1 - s.phi^2) * randn();
            y(1) = exp(h(1) / 2) * randn();
            for t = 2:T
                h(t) = s.mu + s.phi * (h(t-1) - s.mu) + s.sigma_eta * randn();
                y(t) = exp(h(t) / 2) * randn();
            end
        end

        %% ---- Packing ---------------------------------------------

        function s = unpack(~, theta)
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta];
        end
    end
end
