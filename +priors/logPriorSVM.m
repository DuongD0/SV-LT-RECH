function lp = logPriorSVM(theta)
% logPriorSVM  Log prior density for SV-in-mean (Koopman & Hol Uspensky 2002).
%
%   theta = [mu, phi, sigma_eta, alpha0, lambda]
%
%   Priors (PROPOSED_METHODOLOGY.md §9.1 + §7.5d)
%   ---------------------------------------------
%     mu        ~ N(0, 10^2)                 diffuse unconditional log-vol
%     (1+phi)/2 ~ Beta(20, 1.5)              persistence near 0.95
%     sigma_eta ~ Half-Cauchy(0, 1)          vol-of-vol
%     alpha0    ~ N(0, 1)                    return intercept
%     lambda    ~ N(0, 1)                    variance-in-mean / risk-premium
%
%   The N(0,1) priors on (alpha0, lambda) are weakly informative on the scale
%   of 100*log returns (so that lambda*exp(h_t) is O(1) when volatility is
%   O(1%)). Returns -Inf if any parameter is out of support.

    mu        = theta(1);
    phi       = theta(2);
    sigma_eta = theta(3);
    alpha0    = theta(4);
    lambda    = theta(5);

    %% Support checks (mean-equation coefficients are unconstrained reals)
    if phi <= -1 || phi >= 1 || sigma_eta <= 0
        lp = -Inf;
        return
    end

    %% SV core (mu, phi, sigma_eta) shares plain-SV prior.
    lp_sv = priors.logPriorSV([mu, phi, sigma_eta]);

    %% alpha0 ~ N(0, 1), lambda ~ N(0, 1)
    lp_alpha0 = -0.5 * alpha0^2 - 0.5 * log(2 * pi);
    lp_lambda = -0.5 * lambda^2 - 0.5 * log(2 * pi);

    lp = lp_sv + lp_alpha0 + lp_lambda;
end
