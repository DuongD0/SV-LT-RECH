classdef SVLTLSTMRECH < models.Model
% SVLTLSTMRECH  Co-primary SV-LT-LSTM-RECH model (PROPOSED_METHODOLOGY.md §8.4).
%
%   Identical SV backbone to SVLTRECH (§8.2); the RNN cell is an LSTM
%   (hidden dim 1) instead of the ReLU SRN.
%
%   Observation:  r_t = exp(h_t/2) * eps_t,    eps_t ~ scaled t_nu
%   Latent:       h_t = mu + phi(h_{t-1} - mu) + omega_t + sigma_eta * eta_t
%                 corr(eps_{t-1}, eta_t) = rho
%   RNN cell:     omega_t = beta_0 + beta_1 * s_t
%                 [s_t, c_t] = LSTM(x_t, s_{t-1}, c_{t-1})      (cells.lstm)
%                 x_t = [h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1}]
%                 s_1 == 0, c_1 == 0
%
%   theta order (length 15 + 4*nInputs, nInputs = 3 + nCovariates):
%     [mu, phi, sigma_eta, rho, nu, beta_0, beta_1,
%      <f: W_f(1..nInputs), u_f, b_f>, <i: ...>, <o: ...>, <c: ...>]
%
%   Construction mirrors SVLTRECH:
%     mdl = models.SVLTLSTMRECH()
%     mdl = models.SVLTLSTMRECH('nCovariates', 2, 'covariates', Z)
%     mdl = models.SVLTLSTMRECH('leverage', 'ocsn')

    properties (SetAccess = private)
        nCovariates (1,1) double
        leverage    (1,:) char
        couplingFn  function_handle
    end

    properties (SetAccess = public)
        Z (:,:) double = []
    end

    methods

        function obj = SVLTLSTMRECH(opts)
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
                'SVLTLSTMRECH:covariateShape', ...
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
            gates    = {'f', 'i', 'o', 'c'};
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
            lp = priors.logPriorSVLTLSTMRECH(theta, obj.nCovariates);
        end

        function theta = samplePrior(obj, n)
            mu        = 10 * randn(n, 1);
            phi       = 2 * betarnd(20, 1.5, n, 1) - 1;
            sigma_eta = abs(trnd(1, n, 1));
            rho       = 2 * rand(n, 1) - 1;
            nu        = 2 + exprnd(10, n, 1);
            beta_0    = 0.5 * rand(n, 1);
            beta_1    = 0.5 * rand(n, 1);

            nGateW = 4 * (3 + obj.nCovariates + 2);   % 4 gates x (nInputs + u + b)
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

            [sNew, ctx] = cells.lstm(x, aux.sPrev, aux.cPrev, s.lstm);
            omegaNew    = s.beta_0 + s.beta_1 * sNew;

            eta  = obj.couplingFn(s.rho, aux.epsilonPrev, n);
            hNew = s.mu + s.phi * (hOld - s.mu) + omegaNew + s.sigma_eta * eta;

            auxNew = aux;
            auxNew.sPrev     = sNew;
            auxNew.cPrev     = ctx.c;
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
                'cPrev',       zeros(n, 1), ...
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
            cPrev     = 0;
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
                [sNew, ctx] = cells.lstm(xRow, sPrev, cPrev, s.lstm);
                omegaNew    = s.beta_0 + s.beta_1 * sNew;

                eta  = obj.couplingFn(s.rho, epsPrev, 1);
                h(t) = s.mu + s.phi * (h(t-1) - s.mu) + omegaNew + s.sigma_eta * eta;
                y(t) = exp(h(t) / 2) * sigmaEps * trnd(s.nu);

                sPrev     = sNew;
                cPrev     = ctx.c;
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
            gates = {'f', 'i', 'o', 'c'};
            w     = struct();
            for gi = 1:numel(gates)
                g = gates{gi};
                w.(['W_', g]) = theta(off + 1 : off + nInputs);
                w.(['u_', g]) = theta(off + nInputs + 1);
                w.(['b_', g]) = theta(off + nInputs + 2);
                off = off + nInputs + 2;
            end
            s.lstm = w;
        end

        function theta = pack(~, s)
            theta = [s.mu, s.phi, s.sigma_eta, s.rho, s.nu, s.beta_0, s.beta_1, ...
                     s.lstm.W_f, s.lstm.u_f, s.lstm.b_f, ...
                     s.lstm.W_i, s.lstm.u_i, s.lstm.b_i, ...
                     s.lstm.W_o, s.lstm.u_o, s.lstm.b_o, ...
                     s.lstm.W_c, s.lstm.u_c, s.lstm.b_c];
        end
    end
end
