classdef (Abstract) Model < handle
% Model  Abstract base for every volatility model in this codebase.
%
%   The SMC engine, particle filters, simulators, and tests all interact
%   with concrete models through this interface. Subclasses MUST implement
%   every Abstract method below. Two non-abstract hooks (`initParticleAux`,
%   `updateParticleAux`) ship with no-op defaults; subclasses override them
%   only when they need stateful per-particle auxiliary information (e.g.
%   the RNN hidden state in SVLTRECH or the lagged return shock in SVLT).
%
%   Parameter convention
%   --------------------
%   `theta` is a row vector. The order of parameters is fixed per model
%   and exposed via `paramNames`. `unpack(theta)` returns a struct with
%   named fields so internal code reads like math.
%
%   Latent-state convention
%   -----------------------
%   `h` is the log-variance state (or whatever latent variable the model
%   uses). It is scalar per particle for the SV-family; tensorised over
%   the M-particle dimension when called from the particle filter.
%
%   Particle-auxiliary convention
%   -----------------------------
%   `aux` is an arbitrary struct (or struct array) carrying anything the
%   model needs beyond `h` to propagate the next step. Plain SV needs
%   nothing; SVLT carries `epsilonPrev` for the leverage coupling;
%   SVLTRECH additionally carries `sPrev` (RNN hidden state) and
%   `omegaPrev` (long-term volatility component).
%
%   See: PROPOSED_METHODOLOGY.md §§7–8 for the model definitions.

    methods (Abstract)

        %% ---- Parameter metadata -----------------------------------

        names = paramNames(obj)
        % names = paramNames() -> cellstr of parameter names in `theta`
        % order. Implemented as a method (not a Constant property) so
        % that subclasses with construction-dependent parameter counts
        % (SVLTRECH, where the number of covariates varies) can compute
        % the list at runtime. Fixed-parameter models (SV, SVt, SVLT)
        % just return a literal cellstr.

        %% ---- Priors -----------------------------------------------

        lp = logPrior(obj, theta)
        % lp = logPrior(theta) -> scalar log p(theta). -Inf out of support.

        theta = samplePrior(obj, n)
        % Returns an n-by-|paramNames| matrix of prior draws.

        %% ---- Latent state ----------------------------------------

        h0 = initLatent(obj, theta, n)
        % h0 = initLatent(theta, n)
        % Initial latent state for n particles. SV-family typically draws
        % from the stationary distribution h_1 ~ N(mu, sigma_eta^2/(1-phi^2)).

        [hNew, auxNew] = transitionSample(obj, hOld, theta, t, n, aux)
        % [hNew, auxNew] = transitionSample(hOld, theta, t, n, aux)
        % Sample h_t given h_{t-1}. Vectorised over n particles. `aux`
        % carries optional per-particle auxiliary state from the filter
        % (e.g. previous-step return shock for leverage, RNN-derived
        % omega for SVLTRECH). The method returns the updated aux so
        % subclasses that compute intermediate quantities (RNN hidden
        % state, long-term omega) can persist them for the next step.
        % Subclasses that do not use aux should return it unchanged.

        %% ---- Observation -----------------------------------------

        logp = observationLogLik(obj, yT, hT, theta)
        % logp = observationLogLik(yT, hT, theta) -> log p(y_t | h_t, theta).
        % Vectorised: hT is n-by-1, returns n-by-1.

        %% ---- DGP -------------------------------------------------

        [y, h] = simulate(obj, theta, T, varargin)
        % [y, h] = simulate(theta, T) -> length-T return path and latent path.
        % Used by simulator-round-trip tests (§11.1).

        %% ---- Parameter packing -----------------------------------

        s = unpack(obj, theta)
        % s = unpack(theta) -> struct with paramNames as fields.

        theta = pack(obj, s)
        % theta = pack(s) -> flat row vector ordered per paramNames.
    end

    methods

        %% ---- Particle auxiliary state (default no-op) ------------

        function aux = initParticleAux(~, ~, n)
        % aux = initParticleAux(theta, n)
        % Default: scalar struct recording the particle count. Concrete
        % models override to seed `epsilonPrev`, RNN hidden state, etc.
            aux = struct('n', n);
        end

        function [aux, ctx] = updateParticleAux(~, aux, ~, ~, ~, ~)
        % [aux, ctx] = updateParticleAux(aux, hCurr, yCurr, theta, t)
        % Default: pass-through. `ctx` is an opaque struct callers may
        % inspect (empty here).
            ctx = struct();
        end

        function n = nParams(obj)
            n = numel(obj.paramNames);
        end

        function tf = inSupport(obj, theta)
            % Default: in-support iff logPrior is finite.
            tf = isfinite(obj.logPrior(theta));
        end
    end
end
