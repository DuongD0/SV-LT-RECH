function out = mseMae(rvSqrt, vHat)
% mseMae  Mean squared and mean absolute error of volatility forecast.
%
%   out = mseMae(rvSqrt, vHat) -> struct with fields .mse, .mae.
%
%   rvSqrt : T_test-by-1 sqrt(RV_t).
%   vHat   : T_test-by-1 forecast standard deviation v_hat_t.
%            (For Student-t innovations: v_hat^2 = nu * sigma_hat^2 / (nu-2).)
%
%   See PROPOSED_METHODOLOGY.md §10.1.

    arguments
        rvSqrt (:,1) double
        vHat   (:,1) double
    end
    e       = rvSqrt - vHat;
    out.mse = mean(e .^ 2);
    out.mae = mean(abs(e));
end
