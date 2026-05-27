function etaNew = leverageCholesky(rho, epsPrev, n)
% leverageCholesky  Direct Gaussian leverage coupling.
%
%   etaNew = leverageCholesky(rho, epsPrev, n)
%
%   Draws the next-period log-variance innovation `eta_t` given the
%   previous-period standardised return shock `epsPrev = eps_{t-1}` and
%   leverage parameter `rho in (-1, 1)`. Uses the standard Cholesky
%   coupling:
%
%       eta_t = rho * eps_{t-1} + sqrt(1 - rho^2) * v_t,   v_t ~ N(0, 1)
%
%   `eps_{t-1}` is treated as if it were the underlying standard-normal
%   shock; this is exact for SV with Gaussian innovations and an
%   approximation under Student-t (the scale-mixture distortion is
%   absorbed into v_t). For nu > 6 the approximation is empirically
%   negligible; for fatter-tailed regimes the OCSN alternative
%   (`utils.leverageOcsn`) dampens the coupling on outlier residuals.
%
%   Inputs
%   ------
%     rho     : scalar in (-1, 1).
%     epsPrev : n-by-1 vector of standardised residuals from t-1.
%     n       : number of particles (must equal numel(epsPrev)).
%
%   Output
%   ------
%     etaNew  : n-by-1 vector of coupled draws.

    arguments
        rho     (1,1) double {mustBeGreaterThan(rho, -1), mustBeLessThan(rho, 1)}
        epsPrev (:,1) double
        n       (1,1) double {mustBePositive}
    end

    assert(numel(epsPrev) == n, ...
        'leverageCholesky:sizeMismatch', ...
        'epsPrev must have length %d, got %d', n, numel(epsPrev));

    v       = randn(n, 1);
    etaNew  = rho * epsPrev + sqrt(1 - rho^2) * v;
end
