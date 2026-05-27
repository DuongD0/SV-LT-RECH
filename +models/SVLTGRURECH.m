classdef SVLTGRURECH < models.Model
% SVLTGRURECH  Sensitivity SV-LT-GRU-RECH model (PROPOSED_METHODOLOGY.md §8.5).
%
%   Identical SV backbone to SVLTRECH (§8.2); the RNN cell is a GRU
%   (hidden dim 1) instead of the ReLU SRN. Unlike the LSTM variant the
%   GRU carries no separate cell state.
%
%   Observation:  r_t = exp(h_t/2) * eps_t,    eps_t ~ scaled t_nu
%   Latent:       h_t = mu + phi(h_{t-1} - mu) + omega_t + sigma_eta * eta_t
%                 corr(eps_{t-1}, eta_t) = rho
%   RNN cell:     omega_t = beta_0 + beta_1 * s_t
%                 s_t = GRU(x_t, s_{t-1})                       (cells.gru)
%                 x_t = [h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1}]
%                 s_1 == 0
%
%   theta order (length 13 + 3*nInputs, nInputs = 3 + nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      <z: W_z(1..nInputs), u_z, b_z>, <r: ...>, <h: ...>]

    properties (SetAccess = private)
        nCovariates (1,1) double
        leverage    (1,:) char
        couplingFn  function_handle
    end

    properties (SetAccess = public)
        Z (:,:) double = []
    end

    methods

        function obj = SVLTGRURECH(opts)
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
                'SVLTGRURECH:covariateShape', ...
                'Z has %d columns; model expects %d', ...
                size(Z, 2), obj.nCovariates);
            obj.Z = Z;
        end

        %% ---- Metadata --------------------------------------------

        function names = paramNames(obj)
            core     = {'mu', 'phi', 'sigma_eta', 'rho', 'nu', 'beta_0', 'beta_1'};
            inLabels = [{'h', 'r', 'omega'}, ...
                        arrayfun(@(k) sprintf('z%d', k), 1:obj.nCovariates, ...
                                 'UniformOutput', false)];
            gates    = {'z', 'r', 'h'};
            gateNames = {};
            for gi = 1:numel(gates)
                g = gates{gi};
                wNames = cellfun(@(lbl) sprintf('W%s_%s', g, lbl), inLabels, ...
                                 'UniformOutput', false);
                gateNames = [gateNames, wNames, {sprintf('u%s', g), sprintf('b%s', g)}]; %#ok<AGROW>
            end
            names = [core, gateNames];
        end

        %% ---- Priors ---------------------------------------------

        function lp = logPrior(obj, theta)
            lp = priors.logPriorSVLTGRURECH(theta, obj.nCovariates);
        end

        function theta = samplePrior(obj, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));
            rho       = 2 * rand(n, 1) - 1;
            nu        = 2 + exprnd(10, n, 1);
            beta_0    = 0.5 * rand(n, 1);
            beta_1    = 0.5 * rand(n, 1);

            nGateW = 3 * (3 + obj.nCovariates + 2);   % 3 gates x (nInputs + u + b)
            gateW  = 0.1 * randn(n, nGateW);

            theta = [mu, phi, sigma_eta, rho, nu, beta_0, beta_1, gateW];
        end

        %% ---- Latent state ---------------------------------------

        function h0 = initLatent(obj, theta, n)
            s = obj.unpack(theta);
            stationarySd = s.sigma_eta / sqrt(1 - s.phi^2);
            h0 = s.mu + stationarySd * randn(n, 1);
        end

        function [hNew, auxNew] = transitionSample(obj, hOld, theta, t, n, aux)
            s = obj.unpack(theta);

            x = [hOld, aux.yPrev * ones(n, 1), aux.omegaPrev];
            if obj.nCovariates > 0
                x = [x, repmat(obj.Z(t - 1, :), n, 1)];
            end

            sNew     = cells.gru(x, aux.sPrev, s.gru);
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
                sNew     = cells.gru(xRow, sPrev, s.gru);
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

        %% ---- Packing --------------------------------------------

        function s = unpack(obj, theta)
            K       = obj.nCovariates;
            nInputs = 3 + K;
            s.mu        = theta(1);
            s.phi       = theta(2);
            s.sigma_eta = theta(3);
            s.rho       = theta(4);
            s.nu        = theta(5);
            s.beta_0    = theta(6);
            s.beta_1    = theta(7);

            off   = 7;
            gates = {'z', 'r', 'h'};
            w     = struct();
            for gi = 1:numel(gates)
                g = gates{gi};
                w.(['W_', g]) = theta(off + 1 : off + nInputs);
                w.(['u_', g]) = theta(off + nInputs + 1);
                w.(['b_', g]) = theta(off + nInputs + 2);
                off = off + nInputs + 2;
            end
            s.gru = w;
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.rho, s.nu, s.beta_0, s.beta_1, ...
                     s.gru.W_z, s.gru.u_z, s.gru.b_z, ...
                     s.gru.W_r, s.gru.u_r, s.gru.b_r, ...
                     s.gru.W_h, s.gru.u_h, s.gru.b_h];
        end
    end
end
