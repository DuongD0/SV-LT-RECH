function [yTrain, yTest, xTrain, xTest, idx] = trainTestSplit(y, x, trainFraction)
% trainTestSplit  Chronological split — NEVER shuffled.
%
%   [yTrain, yTest, xTrain, xTest, idx] = trainTestSplit(y, x, trainFraction)
%
%   y             : T-by-1 return series.
%   x             : T-by-p covariate matrix, possibly [].
%   trainFraction : scalar in (0, 1), default 0.75.
%
%   idx.trainEnd is the index of the last training observation so
%   yTrain = y(1:trainEnd), yTest = y(trainEnd+1:end).
%
%   No shuffling: returns are temporally dependent and shuffling would
%   destroy the volatility-clustering signal we exist to capture (§6).

    arguments
        y                 (:,1) double
        x                       double = []
        trainFraction     (1,1) double {mustBeInRange(trainFraction, 0, 1)} = 0.75
    end

    T        = numel(y);
    trainEnd = floor(T * trainFraction);

    yTrain = y(1:trainEnd);
    yTest  = y(trainEnd+1:end);

    if ~isempty(x)
        assert(size(x, 1) == T, 'trainTestSplit:lengthMismatch', ...
            'x has %d rows but y has %d', size(x, 1), T);
        xTrain = x(1:trainEnd, :);
        xTest  = x(trainEnd+1:end, :);
    else
        xTrain = [];
        xTest  = [];
    end

    idx.trainEnd = trainEnd;
    idx.T        = T;
end
