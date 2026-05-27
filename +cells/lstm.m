function [sNew, ctx] = lstm(x, sPrev, cPrev, weights)
% lstm  LSTM cell forward pass (hidden dim 1).
%
%   [sNew, ctx] = cells.lstm(x, sPrev, cPrev, weights)
%
%   Computes one step of the LSTM cell used in SV-LT-LSTM-RECH
%   (PROPOSED_METHODOLOGY.md §8.4):
%
%       f_t = sigmoid( W_f x_t + u_f s_{t-1} + b_f )      forget gate
%       i_t = sigmoid( W_i x_t + u_i s_{t-1} + b_i )      input gate
%       o_t = sigmoid( W_o x_t + u_o s_{t-1} + b_o )      output gate
%       g_t = tanh   ( W_c x_t + u_c s_{t-1} + b_c )      candidate
%       c_t = f_t .* c_{t-1} + i_t .* g_t                 cell state
%       s_t = o_t .* tanh(c_t)                            hidden state
%
%   Hidden dim is 1, so s and c are scalars per particle and every
%   recurrent weight u_* is a scalar. Vectorised over n particles.
%
%   Inputs:
%     x       : n-by-d   input vector(s) (x_t = [h_{t-1}, r_{t-1}, omega_{t-1}, z_{t-1}])
%     sPrev   : n-by-1   previous hidden state(s)   (zeros for s_1)
%     cPrev   : n-by-1   previous cell state(s)     (zeros for c_1)
%     weights : struct with per-gate fields (g in {f,i,o,c}):
%                 .W_g  1-by-d input weights
%                 .u_g  scalar recurrent weight
%                 .b_g  scalar bias
%
%   Outputs:
%     sNew : n-by-1 new hidden state(s)
%     ctx  : struct with field .c = n-by-1 new cell state(s). The caller
%            threads ctx.c back in as cPrev on the next step.
%
%   References: PROPOSED_METHODOLOGY.md §8.4; SR-SV (Nguyen et al. 2023);
%   RealRECH (2025).

    arguments
        x       (:,:) double
        sPrev   (:,1) double
        cPrev   (:,1) double
        weights (1,1) struct
    end

    n = size(x, 1);
    assert(numel(sPrev) == n, ...
        'lstm:sizeMismatch', ...
        'sPrev length %d must equal x rows %d', numel(sPrev), n);
    assert(numel(cPrev) == n, ...
        'lstm:sizeMismatch', ...
        'cPrev length %d must equal x rows %d', numel(cPrev), n);

    f = sig(x * weights.W_f(:) + weights.u_f * sPrev + weights.b_f);
    i = sig(x * weights.W_i(:) + weights.u_i * sPrev + weights.b_i);
    o = sig(x * weights.W_o(:) + weights.u_o * sPrev + weights.b_o);
    g = tanh(x * weights.W_c(:) + weights.u_c * sPrev + weights.b_c);

    cNew = f .* cPrev + i .* g;
    sNew = o .* tanh(cNew);

    ctx = struct('c', cNew);
end


function y = sig(z)
% Numerically stable logistic sigmoid: 1/(1+exp(-z)) == (1+tanh(z/2))/2.
    y = 0.5 * (1 + tanh(z / 2));
end
