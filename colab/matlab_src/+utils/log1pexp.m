function y = log1pexp(x)
% log1pexp  Numerically stable log(1 + exp(x)) = softplus.
%
%   Four regimes from Maechler (2012) "Accurately Computing log(1 - exp(.))":
%     x <= -37  :  exp(x)               (1 + exp(x) ~= 1 + tiny)
%     x <=  18  :  log1p(exp(x))        (default safe range)
%     x <=  33.3:  x + exp(-x)          (avoid overflow of exp(x))
%     x  >  33.3:  x                    (exp(-x) underflows to 0)

    y = zeros(size(x));

    r1 = x <= -37;
    r2 = (x > -37) & (x <= 18);
    r3 = (x > 18)  & (x <= 33.3);
    r4 = x > 33.3;

    y(r1) = exp(x(r1));
    y(r2) = log1p(exp(x(r2)));
    y(r3) = x(r3) + exp(-x(r3));
    y(r4) = x(r4);
end
