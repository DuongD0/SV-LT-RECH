function etaNew = leverageOcsn(rho, epsPrev, n)
% leverageOcsn  Robust leverage coupling via the OCSN (2007) 10-component
%               normal mixture for log(eps^2).
%
%   etaNew = leverageOcsn(rho, epsPrev, n)
%
%   Per-particle procedure:
%     1. u_i      = log(epsPrev_i^2 + 1e-12).
%     2. Mixture posterior: w_{k|i} \propto pi_k * N(u_i; m_k, v_k),
%        normalised across k = 1..10 (Omori-Chib-Shephard-Nakajima 2007
%        *J. Econometrics* 140(2), Table 2 approximating log(chi^2_1)).
%     3. Sample component k_i ~ Categorical(w_{:|i}).
%     4. Effective leverage rho_i = rho / (1 + v_{k_i}); high-variance
%        components (Student-t outliers) shrink rho toward zero.
%     5. eta_t_i = rho_i * epsPrev_i + sqrt(1 - rho_i^2) * v_i,
%        v_i ~ N(0, 1).
%
%   Rationale
%   ---------
%   The Cholesky coupling treats `epsPrev` as exactly N(0, 1). Under
%   Student-t innovations the tail residuals are too large for a
%   Gaussian, and Cholesky over-couples — driving the proposal for h_t
%   into rare regions and starving the particle filter. The OCSN
%   mixture explicitly recognises which residuals are "tail-component"
%   draws and dampens the coupling for those particles. The bulk of
%   particles (typical residuals concentrated near components k = 4-6)
%   retain near-full leverage; outliers (k = 1-3 or 9-10) get
%   automatic shrinkage.
%
%   See PROPOSED_METHODOLOGY.md §8.2 + phase log §4.2.2 for the
%   broader leverage-coupling discussion. OCSN Table 2 constants are
%   hard-coded below for reproducibility.

    arguments
        rho     (1,1) double {mustBeGreaterThan(rho, -1), mustBeLessThan(rho, 1)}
        epsPrev (:,1) double
        n       (1,1) double {mustBePositive}
    end

    assert(numel(epsPrev) == n, ...
        'leverageOcsn:sizeMismatch', ...
        'epsPrev must have length %d, got %d', n, numel(epsPrev));

    [wPrior, mComp, vComp] = ocsnComponents();      % 1-by-10 each

    u    = log(epsPrev.^2 + 1e-12);
    logW = -0.5 * log(2 * pi * vComp) ...
           - 0.5 * (u - mComp).^2 ./ vComp ...
           + log(wPrior);

    logWmax = max(logW, [], 2);
    W       = exp(logW - logWmax);
    W       = W ./ sum(W, 2);

    cdf  = cumsum(W, 2);
    r    = rand(n, 1);
    kIdx = sum(r > cdf, 2) + 1;
    kIdx = min(kIdx, numel(wPrior));

    vK     = vComp(kIdx)';
    rhoEff = rho ./ (1 + vK);
    noise  = randn(n, 1);
    etaNew = rhoEff .* epsPrev + sqrt(1 - rhoEff.^2) .* noise;
end


function [w, m, v] = ocsnComponents()
% Omori, Chib, Shephard, Nakajima (2007) Table 2 -- 10-component normal
% mixture approximating log(chi^2_1). w = pi_k (weights), m = means,
% v = variances. Hard-coded for reproducibility.
    w = [0.00609, 0.04775, 0.13057, 0.20674, 0.22715, ...
         0.18842, 0.12047, 0.05591, 0.01575, 0.00115];
    m = [ 1.92677,  1.34744,  0.73504,  0.02266, -0.85173, ...
         -1.97278, -3.46788, -5.55246, -8.68384, -14.65000];
    v = [0.11265, 0.17788, 0.26768, 0.40611, 0.62699, ...
         0.98583, 1.57469, 2.54498, 4.16591,  7.33342];
end
