function lp = logPriorSV(theta)
% logPriorSV  Log prior density for plain Stochastic Volatility parameters.
%
%   theta = [mu, phi, sigma_eta]
%
%   Priors (from PROPOSED_METHODOLOGY.md §9.1)
%   -----------------------------------------
%     mu        ~ N(0, 10^2)                       diffuse unconditional log-vol
%     (1+phi)/2 ~ Beta(20, 1.5)                    persistence near 0.95
%     sigma_eta ~ Half-Cauchy(0, 1)                vol-of-vol
%
%   Returns -Inf if any parameter is out of support.

    mu        = theta(1);
    phi       = theta(2);
    sigma_eta = theta(3);

    %% Support checks
    if phi <= -1 || phi >= 1 || sigma_eta <= 0
        lp = -Inf;
        return
    end

    %% mu ~ N(0, 10)
    sd_mu = 10;
    lp_mu = -0.5 * (mu / sd_mu)^2 - log(sd_mu) - 0.5 * log(2 * pi);

    %% (1+phi)/2 ~ Beta(20, 1.5), change of variables: log|Jacobian| = -log(2)
    ph     = (1 + phi) / 2;
    a      = 20;
    b      = 1.5;
    lp_phi = (a - 1) * log(ph) + (b - 1) * log(1 - ph) - betaln(a, b) - log(2);

    %% sigma_eta ~ Half-Cauchy(0, 1)
    lp_sig = log(2) - log(pi) - log(1 + sigma_eta^2);

    lp = lp_mu + lp_phi + lp_sig;
end
