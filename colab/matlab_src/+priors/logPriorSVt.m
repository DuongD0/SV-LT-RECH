function lp = logPriorSVt(theta)
% logPriorSVt  Log prior density for SV + Student-t innovations.
%
%   theta = [mu, phi, sigma_eta, nu]
%
%   Priors (PROPOSED_METHODOLOGY.md §9.1 + phase log §4.2.1)
%   --------------------------------------------------------
%     mu        ~ N(0, 10^2)
%     (1+phi)/2 ~ Beta(20, 1.5)
%     sigma_eta ~ Half-Cauchy(0, 1)
%     nu        ~ Gamma(shape = 1, rate = 0.1)        truncated to nu > 2
%
%   The Gamma prior on nu has mean = 10 and is weakly informative over
%   the empirically relevant range (4-30). We truncate at 2 because the
%   scaled-t observation density requires finite variance (nu > 2).
%
%   Returns -Inf if any parameter is out of support.

    mu        = theta(1);
    phi       = theta(2);
    sigma_eta = theta(3);
    nu        = theta(4);

    if phi <= -1 || phi >= 1 || sigma_eta <= 0 || nu <= 2
        lp = -Inf;
        return
    end

    %% SV core (same as logPriorSV)
    lp_sv = priors.logPriorSV([mu, phi, sigma_eta]);

    %% nu ~ Gamma(shape=1, rate=0.1) truncated to (2, inf).
    % For shape=1 this is Exponential(rate=0.1): log pdf = log(0.1) - 0.1*nu.
    % Truncation at nu>2 divides by 1 - F(2) = exp(-0.2), i.e. add 0.2.
    rate = 0.1;
    lp_nu = log(rate) - rate * nu - (-rate * 2);

    lp = lp_sv + lp_nu;
end
