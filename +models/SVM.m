classdef SVM < models.Model
% SVM  Stochastic Volatility in Mean (Koopman & Hol Uspensky 2002, §7.5d).
%
%   The unobserved volatility enters the MEAN equation of returns, so the
%   model captures the intertemporal risk-return trade-off (volatility
%   feedback / risk premium) that the plain SV family cannot.
%
%   Observation:  r_t = alpha0 + lambda * exp(h_t) + exp(h_t/2) * eps_t,
%                 eps_t ~ N(0, 1)
%
%     * alpha0                : constant return intercept (Koopman's beta_0)
%     * lambda * exp(h_t)     : VARIANCE-in-mean term (Koopman's beta_2 * e^{h_t}).
%                               lambda > 0 => positive volatility-feedback /
%                               risk premium; lambda < 0 => leverage-of-the-mean.
%     * exp(h_t/2) * eps_t     : standard SV scale (Var(r_t | h_t) = exp(h_t)).
%
%   Latent state: h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,
%                 eta_t ~ N(0, 1),  |phi| < 1,  sigma_eta > 0
%
%   Initial state: h_1 ~ N(mu, sigma_eta^2 / (1 - phi^2))   (stationary)
%
%   theta = [mu, phi, sigma_eta, alpha0, lambda]
%
%   Gaussian innovations match Koopman & Hol Uspensky (2002). A heavy-tailed
%   (scaled-t) extension is the SMN generalisation of Abanto-Valle et al.
%   (2017); it is intentionally NOT enabled here so that SVM pairs cleanly
%   with the Gaussian plain-SV baseline for the volatility-in-mean test.
%
%   References: Koopman & Hol Uspensky (2002) J. Appl. Econometrics 17(6);
%   Abanto-Valle, Lachos & Dey (2017); PROPOSED_METHODOLOGY.md §7.5d.

    methods

        function names = paramNames(~)
            names = {'mu', 'phi', 'sigma_eta', 'alpha0', 'lambda'};
        end

        %% ---- Priors -----------------------------------------------

        function lp = logPrior(~, theta)
            lp = priors.logPriorSVM(theta);
        end

        function theta = samplePrior(~, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));          % half-Cauchy
            alpha0    = randn(n, 1);                 % N(0, 1)
            lambda    = randn(n, 1);                 % N(0, 1) in-mean coef
            theta     = [mu, phi, sigma_eta, alpha0, lambda];
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
            % log N(y_t; alpha0 + lambda*exp(h_t), exp(h_t)). Vectorised
            % over the M-by-1 particle dimension hT (yT is scalar).
            s     = obj.unpack(theta);
            meanT = s.alpha0 + s.lambda .* exp(hT);
            logp  = -0.5 * log(2 * pi) ...
                  - 0.5 * hT ...
                  - 0.5 * (yT - meanT).^2 .* exp(-hT);
        end

        %% ---- DGP --------------------------------------------------

        function [y, h] = simulate(obj, theta, T, varargin)
            s = obj.unpack(theta);
            h = zeros(T, 1);
            y = zeros(T, 1);
            h(1) = s.mu + s.sigma_eta / sqrt(1 - s.phi^2) * randn();
            y(1) = s.alpha0 + s.lambda * exp(h(1)) + exp(h(1) / 2) * randn();
            for t = 2:T
                h(t) = s.mu + s.phi * (h(t-1) - s.mu) + s.sigma_eta * randn();
                y(t) = s.alpha0 + s.lambda * exp(h(t)) + exp(h(t) / 2) * randn();
            end
        end

        %% ---- Packing ---------------------------------------------

        function s = unpack(~, theta)
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
            s.alpha0    = theta(4);
            s.lambda    = theta(5);
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.alpha0, s.lambda];
        end
    end
end
