function out = bdsTest(x, m, epsFrac)
% bdsTest  Brock-Dechert-Scheinkman test for iid (nonlinear dependence).
%
%   out = bdsTest(x)
%   out = bdsTest(x, m, epsFrac)
%
%   Tests H0: x is iid against unspecified (often nonlinear) dependence,
%   via the correlation-integral statistic of Brock, Dechert, Scheinkman &
%   LeBaron (1996). Rejection on residuals motivates the neural-network
%   correction (PROPOSED_METHODOLOGY.md §4, Table T1).
%
%   Inputs
%   ------
%     x       : T-by-1 series (e.g. returns or standardized residuals).
%     m       : embedding dimension (default 2).
%     epsFrac : neighbourhood radius as a fraction of std(x) (default 0.5).
%
%   Output (struct `out`)
%   ---------------------
%     .stat   : standardized BDS statistic (~ N(0,1) under H0).
%     .pValue : two-sided normal p-value.
%     .m, .eps: echoed settings.

    arguments
        x       (:,1) double
        m       (1,1) double {mustBeInteger, mustBeGreaterThanOrEqual(m, 2)} = 2
        epsFrac (1,1) double {mustBePositive} = 0.5
    end

    T   = numel(x);
    eps = epsFrac * std(x);

    % Pairwise indicator A(s,t) = 1 if |x_s - x_t| < eps (T-by-T, symmetric).
    A = abs(x - x.') < eps;

    % m=1 correlation integral over ordered distinct pairs.
    g  = sum(A, 2) - 1;                 % per-row neighbour count (exclude self)
    C1 = sum(g) / (T * (T - 1));

    % K: probability three points are pairwise within eps.
    K = (sum(g.^2) - sum(g)) / (T * (T - 1) * (T - 2));

    % m-dimensional correlation integral on overlapping m-histories.
    M  = T - m + 1;
    Am = true(M, M);
    for j = 0:(m - 1)
        Am = Am & A(1 + j : M + j, 1 + j : M + j);
    end
    Cm = (sum(Am(:)) - M) / (M * (M - 1));   % exclude diagonal; ordered pairs

    % BDS variance (Brock et al. 1996, eq. for dimension m).
    sigma2 = K^m + (m - 1)^2 * C1^(2 * m) - m^2 * K * C1^(2 * m - 2);
    for j = 1:(m - 1)
        sigma2 = sigma2 + 2 * (K^(m - j) * C1^(2 * j));
    end
    sigma2 = 4 * sigma2;

    out.stat   = sqrt(T) * (Cm - C1^m) / sqrt(max(sigma2, 1e-12));
    out.pValue = 2 * (1 - normcdf(abs(out.stat)));
    out.m      = m;
    out.eps    = eps;
end
