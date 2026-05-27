classdef SVLT < models.Model
% SVLT  Stochastic volatility with leverage + Student-t (§7.5c).
%
%   Observation:  r_t = exp(h_t/2) * eps_t,
%                 eps_t = sqrt((nu-2)/nu) * T_nu        Var(eps_t) = 1
%
%   Latent state: h_t = mu + phi (h_{t-1} - mu) + sigma_eta * eta_t,
%                 corr(eps_{t-1}, eta_t) = rho           (leverage)
%
%   theta = [mu, phi, sigma_eta, rho, nu]
%
%   Leverage coupling strategies
%   ----------------------------
%   Construct as
%       mdl = models.SVLT()                          % default Cholesky
%       mdl = models.SVLT('leverage', 'cholesky')
%       mdl = models.SVLT('leverage', 'ocsn')
%
%   The Cholesky coupling treats eps_{t-1} as standard-normal exactly;
%   the OCSN coupling dampens leverage on Student-t outliers via the
%   Omori-Chib-Shephard-Nakajima (2007) 10-component mixture. See
%   utils.leverageCholesky / utils.leverageOcsn and phase log §4.2.2.

    properties (SetAccess = private)
        leverage   (1,:) char
        couplingFn function_handle
    end

    methods

        function names = paramNames(~)
            names = {'mu', 'phi', 'sigma_eta', 'rho', 'nu'};
        end

        function obj = SVLT(opts)
            arguments
                opts.leverage (1,:) char ...
                    {mustBeMember(opts.leverage, {'cholesky','ocsn'})} = 'cholesky'
            end
            obj.leverage = opts.leverage;
            switch obj.leverage
                case 'cholesky'
                    obj.couplingFn = @utils.leverageCholesky;
                case 'ocsn'
                    obj.couplingFn = @utils.leverageOcsn;
            end
        end

        %% ---- Priors -----------------------------------------------

        function lp = logPrior(~, theta)
            lp = priors.logPriorSVLT(theta);
        end

        function theta = samplePrior(~, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));
            rho       = 2 * rand(n, 1) - 1;
            nu        = 2 + exprnd(10, n, 1);
            theta     = [mu, phi, sigma_eta, rho, nu];
        end

        %% ---- Latent state ----------------------------------------

        function h0 = initLatent(obj, theta, n)
            s = obj.unpack(theta);
            stationarySd = s.sigma_eta / sqrt(1 - s.phi^2);
            h0 = s.mu + stationarySd * randn(n, 1);
        end

        function [hNew, auxNew] = transitionSample(obj, hOld, theta, ~, n, aux)
            s      = obj.unpack(theta);
            eta    = obj.couplingFn(s.rho, aux.epsilonPrev, n);
            hNew   = s.mu + s.phi * (hOld - s.mu) + s.sigma_eta * eta;
            auxNew = aux;
        end

        %% ---- Observation -----------------------------------------

        function logp = observationLogLik(obj, yT, hT, theta)
            % Same scaled-t density as SVt (leverage enters via the
            % transition equation, not the observation).
            s  = obj.unpack(theta);
            nu = s.nu;
            scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
            logp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
                 - 0.5 * log(pi * (nu - 2)) ...
                 - 0.5 * hT ...
                 - ((nu + 1) / 2) * log1p(scaledSq);
        end

        %% ---- Particle auxiliary --------------------------------

        function aux = initParticleAux(~, ~, n)
            % epsilonPrev is overwritten at t=1's updateParticleAux call
            % before being read at t=2's transitionSample.
            aux = struct('epsilonPrev', zeros(n, 1));
        end

        function [aux, ctx] = updateParticleAux(~, aux, hCurr, yCurr, ~, ~)
            aux.epsilonPrev = yCurr ./ exp(hCurr / 2);
            ctx = struct();
        end

        %% ---- DGP --------------------------------------------------

        function [y, h] = simulate(obj, theta, T, varargin)
            s        = obj.unpack(theta);
            sigmaEps = sqrt((s.nu - 2) / s.nu);
            h = zeros(T, 1);
            y = zeros(T, 1);

            h(1)    = s.mu + s.sigma_eta / sqrt(1 - s.phi^2) * randn();
            y(1)    = exp(h(1) / 2) * sigmaEps * trnd(s.nu);
            epsPrev = y(1) / exp(h(1) / 2);

            for t = 2:T
                eta     = obj.couplingFn(s.rho, epsPrev, 1);
                h(t)    = s.mu + s.phi * (h(t-1) - s.mu) + s.sigma_eta * eta;
                y(t)    = exp(h(t) / 2) * sigmaEps * trnd(s.nu);
                epsPrev = y(t) / exp(h(t) / 2);
            end
        end

        %% ---- Packing ---------------------------------------------

        function s = unpack(~, theta)
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
            s.rho       = theta(4);
            s.nu        = theta(5);
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.rho, s.nu];
        end
    end
end
