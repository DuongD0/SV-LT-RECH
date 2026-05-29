classdef SVLTRECH < models.Model
% SVLTRECH  Flagship SV-LT-RECH model (PROPOSED_METHODOLOGY.md §8.2).
%
%   Observation:  r_t = exp(h_t/2) * eps_t,    eps_t ~ scaled t_nu
%   Latent:       h_t = mu + phi(h_{t-1} - mu) + omega_t + sigma_eta * eta_t
%                 corr(eps_{t-1}, eta_t) = rho
%   RNN cell:     omega_t = beta_0 + beta_1 * s_t
%                 s_t = ReLU( v_h h_{t-1} + v_r r_{t-1} + v_omega omega_{t-1}
%                             + v_z^T z_{t-1} + w_h s_{t-1} + b )
%                 s_1 == 0
%
%   theta order (length 12 + K, K = nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      v_h, v_r, v_omega, v_z(1)..v_z(K), w_h, b]
%
%   Construction
%   ------------
%     mdl = models.SVLTRECH()                          % no covariates
%     mdl = models.SVLTRECH('nCovariates', 2)
%     mdl = models.SVLTRECH('nCovariates', 2, 'covariates', Z)
%     mdl = models.SVLTRECH('leverage', 'ocsn')
%
%   Z is bound to the model so SMC / PF stays one-line; replace via
%   `mdl.setCovariates(Z)`. See PROPOSED_METHODOLOGY.md §8.2 and phase
%   log §4.2.3.

    properties (SetAccess = private)
        nCovariates (1,1) double
        leverage    (1,:) char
        couplingFn  function_handle
    end

    properties (SetAccess = public)
        Z (:,:) double = []
    end

    methods

        function obj = SVLTRECH(opts)
            arguments
                opts.nCovariates (1,1) double {mustBeNonnegative, mustBeInteger} = 0
                opts.covariates  (:,:) double = []
                opts.leverage    (1,:) char ...
                    {mustBeMember(opts.leverage, {'cholesky','ocsn'})} = 'cholesky'
            end
            obj.nCovariates = opts.nCovariates;
            obj.leverage    = opts.leverage;
            switch obj.leverage
                case 'cholesky', obj.couplingFn = @utils.leverageCholesky;
                case 'ocsn',     obj.couplingFn = @utils.leverageOcsn;
            end
            if ~isempty(opts.covariates)
                obj.setCovariates(opts.covariates);
            end
        end

        function setCovariates(obj, Z)
            arguments
                obj
                Z (:,:) double
            end
            assert(size(Z, 2) == obj.nCovariates, ...
                'SVLTRECH:covariateShape', ...
                'Z has %d columns; model expects %d', ...
                size(Z, 2), obj.nCovariates);
            obj.Z = Z;
        end

        %% ---- Metadata --------------------------------------------

        function names = paramNames(obj)
            base = {'mu', 'phi', 'sigma_eta', 'rho', 'nu', ...
                    'beta_0', 'beta_1', 'v_h', 'v_r', 'v_omega'};
            cov  = arrayfun(@(k) sprintf('v_z_%d', k), 1:obj.nCovariates, ...
                            'UniformOutput', false);
            tail = {'w_h', 'b'};
            names = [base, cov, tail];
        end

        %% ---- Priors ---------------------------------------------

        function lp = logPrior(obj, theta)
            lp = priors.logPriorSVLTRECH(theta, obj.nCovariates);
        end

        function theta = samplePrior(obj, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));
            rho       = 2 * rand(n, 1) - 1;
            nu        = 2 + exprnd(10, n, 1);
            beta_0    = 0.5 * rand(n, 1);
            beta_1    = 0.5 * rand(n, 1);
            v_h       = 0.1 * randn(n, 1);
            v_r       = 0.1 * randn(n, 1);
            v_omega   = 0.1 * randn(n, 1);
            v_z       = 0.5 * randn(n, obj.nCovariates);
            w_h       = 0.1 * randn(n, 1);
            b         = 0.1 * randn(n, 1);

            theta = [mu, phi, sigma_eta, rho, nu, beta_0, beta_1, ...
                     v_h, v_r, v_omega, v_z, w_h, b];
        end

        %% ---- Latent state ---------------------------------------

        function h0 = initLatent(obj, theta, n)
            s = obj.unpack(theta);
            stationarySd = s.sigma_eta / sqrt(1 - s.phi^2);
            h0 = s.mu + stationarySd * randn(n, 1);
        end

        function [hNew, auxNew] = transitionSample(obj, hOld, theta, t, n, aux)
            s = obj.unpack(theta);

            % Build RNN input x_t per particle: [h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1}]
            x = [hOld, aux.yPrev * ones(n, 1), aux.omegaPrev];
            if obj.nCovariates > 0
                zRow = obj.Z(t - 1, :);
                x    = [x, repmat(zRow, n, 1)];
            end

            weights = struct('v',   [s.v_h, s.v_r, s.v_omega, s.v_z(:)'], ...
                             'w_h', s.w_h, ...
                             'b',   s.b);
            sNew     = cells.srn(x, aux.sPrev, weights);
            omegaNew = s.beta_0 + s.beta_1 * sNew;

            eta  = obj.couplingFn(s.rho, aux.epsilonPrev, n);
            hNew = s.mu + s.phi * (hOld - s.mu) + omegaNew + s.sigma_eta * eta;

            auxNew = aux;
            auxNew.sPrev     = sNew;
            auxNew.omegaPrev = omegaNew;
        end

        %% ---- Observation ----------------------------------------

        function logp = observationLogLik(obj, yT, hT, theta)
            s  = obj.unpack(theta);
            nu = s.nu;
            scaledSq = yT.^2 ./ (exp(hT) * (nu - 2));
            logp = gammaln((nu + 1) / 2) - gammaln(nu / 2) ...
                 - 0.5 * log(pi * (nu - 2)) ...
                 - 0.5 * hT ...
                 - ((nu + 1) / 2) * log1p(scaledSq);
        end

        %% ---- Particle auxiliary ---------------------------------

        function aux = initParticleAux(obj, theta, n)
            s = obj.unpack(theta);
            aux = struct( ...
                'epsilonPrev', zeros(n, 1), ...
                'yPrev',       0, ...
                'sPrev',       zeros(n, 1), ...
                'omegaPrev',   s.beta_0 * ones(n, 1));
        end

        function [aux, ctx] = updateParticleAux(~, aux, hCurr, yCurr, ~, ~)
            aux.epsilonPrev = yCurr ./ exp(hCurr / 2);
            aux.yPrev       = yCurr;
            ctx = struct();
        end

        %% ---- DGP -------------------------------------------------

        function [y, h] = simulate(obj, theta, T, varargin)
            s        = obj.unpack(theta);
            sigmaEps = sqrt((s.nu - 2) / s.nu);
            h = zeros(T, 1);
            y = zeros(T, 1);

            h(1) = s.mu + s.sigma_eta / sqrt(1 - s.phi^2) * randn();
            y(1) = exp(h(1) / 2) * sigmaEps * trnd(s.nu);

            sPrev     = 0;
            omegaPrev = s.beta_0;
            epsPrev   = y(1) / exp(h(1) / 2);
            yPrev     = y(1);

            for t = 2:T
                if obj.nCovariates > 0
                    zRow = obj.Z(t - 1, :);
                else
                    zRow = [];
                end
                xRow = [h(t-1), yPrev, omegaPrev, zRow];
                w    = struct('v',   [s.v_h, s.v_r, s.v_omega, s.v_z(:)'], ...
                              'w_h', s.w_h, ...
                              'b',   s.b);
                sNew     = cells.srn(xRow, sPrev, w);
                omegaNew = s.beta_0 + s.beta_1 * sNew;

                eta  = obj.couplingFn(s.rho, epsPrev, 1);
                h(t) = s.mu + s.phi * (h(t-1) - s.mu) + omegaNew + s.sigma_eta * eta;
                y(t) = exp(h(t) / 2) * sigmaEps * trnd(s.nu);

                sPrev     = sNew;
                omegaPrev = omegaNew;
                epsPrev   = y(t) / exp(h(t) / 2);
                yPrev     = y(t);
            end
        end

        %% ---- Interpretability ------------------------------------

        function omega = omegaPath(obj, theta, hPath, yPath)
        % omega = omegaPath(theta, hPath, yPath)
        % Deterministic recurrent-state path omega_t given a parameter
        % vector and a (filtered) latent-/return-path. Replays the same SRN
        % recurrence as `simulate`. omega_1 == beta_0 (s_1 == 0). For F4.
            arguments
                obj
                theta (1,:) double
                hPath (:,1) double
                yPath (:,1) double
            end
            s = obj.unpack(theta);
            T = numel(hPath);
            omega = zeros(T, 1);
            omega(1) = s.beta_0;
            sPrev = 0; omegaPrev = s.beta_0;
            for t = 2:T
                if obj.nCovariates > 0
                    zRow = obj.Z(t - 1, :);
                else
                    zRow = [];
                end
                xRow = [hPath(t - 1), yPath(t - 1), omegaPrev, zRow];
                w    = struct('v',   [s.v_h, s.v_r, s.v_omega, s.v_z(:)'], ...
                              'w_h', s.w_h, 'b', s.b);
                sNew      = cells.srn(xRow, sPrev, w);
                omega(t)  = s.beta_0 + s.beta_1 * sNew;
                sPrev     = sNew;
                omegaPrev = omega(t);
            end
        end

        %% ---- Packing --------------------------------------------

        function s = unpack(obj, theta)
            K = obj.nCovariates;
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
            s.rho       = theta(4);
            s.nu        = theta(5);
            s.beta_0    = theta(6);
            s.beta_1    = theta(7);
            s.v_h       = theta(8);
            s.v_r       = theta(9);
            s.v_omega   = theta(10);
            if K > 0
                s.v_z = theta(11 : 10 + K);
            else
                s.v_z = zeros(1, 0);
            end
            s.w_h = theta(11 + K);
            s.b   = theta(12 + K);
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.rho, s.nu, ...
                     s.beta_0, s.beta_1, ...
                     s.v_h, s.v_r, s.v_omega, ...
                     s.v_z(:)', s.w_h, s.b];
        end
    end
end
