function [sigma2, omegaPath] = rechCondVar(P, y, Z, cellType, seedVar)
% rechCondVar  Deterministic GARCH-RECH conditional-variance recursion.
%
%   [sigma2, omegaPath] = garch.rechCondVar(P, y, Z, cellType, seedVar)
%
%   Single source of truth for the RECH(1,1) recursion shared by
%   garch.fitGarchRECH (likelihood) and garch.rollingForecastRECH (OOS roll):
%
%       sigma2_1 = seedVar,  omega_1 = beta_0,  s_1 = 0
%       x_t      = [ sigma2_{t-1}, y_{t-1}, omega_{t-1}, z_{t-1} ]
%       s_t      = RNNcell(x_t, s_{t-1})
%       omega_t  = beta_0 + beta_1 * s_t
%       sigma2_t = max( omega_t + alpha*y_{t-1}^2 + beta*sigma2_{t-1}, 1e-8 )
%
%   Inputs
%   ------
%     P        : struct with fields alpha, beta, beta0, beta1, cellWeights
%                (cellWeights in the format expected by cells.<cellType>).
%     y        : T-by-1 return series (mean-zero).
%     Z        : T-by-K covariate matrix or [] (K = 0).
%     cellType : 'srn' | 'lstm' | 'gru'.
%     seedVar  : scalar sigma2_1 seed (unconditional variance).
%
%   Outputs
%   -------
%     sigma2    : T-by-1 conditional-variance path (1-step-ahead from t-1).
%     omegaPath : T-by-1 RNN-driven omega_t path.
%
%   Because sigma2_t depends only on information through t-1, the slice of
%   sigma2 over a held-out window is a genuine 1-step-ahead forecast path.
%
%   See PROPOSED_METHODOLOGY.md §7.6 and +models/SVLTRECH.m (SV analogue).

    arguments
        P        (1,1) struct
        y        (:,1) double
        Z        (:,:) double
        cellType (1,:) char
        seedVar  (1,1) double
    end

    T         = numel(y);
    sigma2    = zeros(T, 1);
    omegaPath = zeros(T, 1);
    sigma2(1)    = max(seedVar, 1e-8);
    omegaPath(1) = P.beta0;

    sPrev     = 0;          % s_1 == 0
    cPrev     = 0;          % LSTM cell state (unused by srn/gru)
    omegaPrev = P.beta0;

    for t = 2:T
        if isempty(Z)
            x = [sigma2(t-1), y(t-1), omegaPrev];
        else
            x = [sigma2(t-1), y(t-1), omegaPrev, Z(t-1, :)];
        end

        switch cellType
            case 'srn'
                sNew = cells.srn(x, sPrev, P.cellWeights);
            case 'lstm'
                [sNew, ctx] = cells.lstm(x, sPrev, cPrev, P.cellWeights);
                cPrev = ctx.c;
            case 'gru'
                sNew = cells.gru(x, sPrev, P.cellWeights);
            otherwise
                error('rechCondVar:badCell', 'Unknown cell %s', cellType);
        end

        omegaT       = P.beta0 + P.beta1 * sNew;
        sigma2(t)    = max(omegaT + P.alpha * y(t-1)^2 + P.beta * sigma2(t-1), 1e-8);
        omegaPath(t) = omegaT;

        sPrev     = sNew;
        omegaPrev = omegaT;
    end
end
