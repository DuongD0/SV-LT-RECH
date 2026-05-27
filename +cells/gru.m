function [sNew, ctx] = gru(x, sPrev, weights)
% gru  GRU cell forward pass (hidden dim 1).
%
%   [sNew, ctx] = cells.gru(x, sPrev, weights)
%
%   Computes one step of the GRU cell used in SV-LT-GRU-RECH
%   (PROPOSED_METHODOLOGY.md §8.5):
%
%       z_t  = sigmoid( W_z x_t + u_z s_{t-1} + b_z )         update gate
%       r_t  = sigmoid( W_r x_t + u_r s_{t-1} + b_r )         reset gate
%       g_t  = tanh   ( W_h x_t + u_h (r_t .* s_{t-1}) + b_h ) candidate
%       s_t  = (1 - z_t) .* s_{t-1} + z_t .* g_t              hidden state
%
%   Hidden dim is 1, so s is scalar per particle and every recurrent
%   weight u_* is a scalar. Vectorised over n particles. Unlike the LSTM
%   cell, the GRU carries no separate cell state, so ctx is empty.
%
%   Inputs:
%     x       : n-by-d   input vector(s) (x_t = [h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1}])
%     sPrev   : n-by-1   previous hidden state(s)   (zeros for s_1)
%     weights : struct with per-gate fields (g in {z,r,h}):
%                 .W_g  1-by-d input weights
%                 .u_g  scalar recurrent weight
%                 .b_g  scalar bias
%
%   Outputs:
%     sNew : n-by-1 new hidden state(s)
%     ctx  : struct, empty (GRU has no separate cell state to carry).
%
%   References: PROPOSED_METHODOLOGY.md §8.5; Cho et al. (2014).

    arguments
        x       (:,:) double
        sPrev   (:,1) double
        weights (1,1) struct
    end

    n = size(x, 1);
    assert(numel(sPrev) == n, ...
        'gru:sizeMismatch', ...
        'sPrev length %d must equal x rows %d', numel(sPrev), n);

    zg = sig(x * weights.W_z(:) + weights.u_z * sPrev + weights.b_z);
    rg = sig(x * weights.W_r(:) + weights.u_r * sPrev + weights.b_r);
    g  = tanh(x * weights.W_h(:) + weights.u_h * (rg .* sPrev) + weights.b_h);

    sNew = (1 - zg) .* sPrev + zg .* g;
    ctx  = struct();
end


function y = sig(z)
% Numerically stable logistic sigmoid: 1/(1+exp(-z)) == (1+tanh(z/2))/2.
    y = 0.5 * (1 + tanh(z / 2));
end
