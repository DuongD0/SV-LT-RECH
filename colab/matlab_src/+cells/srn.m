function [sNew, ctx] = srn(x, sPrev, weights)
% srn  ReLU simple-recurrent-network cell forward pass (hidden dim 1).
%
%   [sNew, ctx] = cells.srn(x, sPrev, weights)
%
%   Computes one step of the RECH-style simple-RNN cell:
%
%       s_t = ReLU( v^T x_t + w_h * s_{t-1} + b )
%
%   Vectorised over n particles. Inputs:
%     x       : n-by-d   input vector(s)
%     sPrev   : n-by-1   previous hidden state(s) (use zeros for s_1)
%     weights : struct with fields
%                 .v   1-by-d input weights
%                 .w_h scalar recurrent weight
%                 .b   scalar bias
%
%   Outputs:
%     sNew : n-by-1 new hidden state(s)
%     ctx  : struct, empty (only LSTM/GRU need to carry cell state)
%
%   References: PROPOSED_METHODOLOGY.md §8.2; Nguyen et al. (2022) RECH
%   Eq. (3); phase log §4.2.3.

    arguments
        x       (:,:) double
        sPrev   (:,1) double
        weights (1,1) struct
    end

    n = size(x, 1);
    assert(numel(sPrev) == n, ...
        'srn:sizeMismatch', ...
        'sPrev length %d must equal x rows %d', numel(sPrev), n);

    z    = x * weights.v(:) + weights.w_h * sPrev + weights.b;
    sNew = max(z, 0);
    ctx  = struct();
end
