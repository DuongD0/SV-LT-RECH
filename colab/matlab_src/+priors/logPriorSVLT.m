function lp = logPriorSVLT(theta)
% logPriorSVLT  Log prior density for SV + leverage + Student-t.
%
%   theta = [mu, phi, sigma_eta, rho, nu]
%
%   Priors (PROPOSED_METHODOLOGY.md §9.1 + phase log §4.2.2)
%   --------------------------------------------------------
%     mu        ~ N(0, 10^2)
%     (1+phi)/2 ~ Beta(20, 1.5)
%     sigma_eta ~ Half-Cauchy(0, 1)
%     rho       ~ Uniform(-1, 1)
%     nu        ~ Gamma(shape = 1, rate = 0.1)  truncated to nu > 2
%
%   Returns -Inf if any parameter is out of support.

    rho = theta(4);

    if rho <= -1 || rho >= 1
        lp = -Inf;
        return
    end

    lp_svt = priors.logPriorSVt([theta(1:3), theta(5)]);
    lp_rho = -log(2);

    lp = lp_svt + lp_rho;
end
