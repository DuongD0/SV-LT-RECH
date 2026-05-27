function [trainStd, testStd, stats] = standardize(train, test)
% standardize  Leakage-safe standardisation using IN-SAMPLE statistics only.
%
%   [trainStd, testStd, stats] = standardize(train, test)
%
%   For each column j:
%       mu_j = mean(train(:, j))                 % NaN-aware
%       sd_j = std (train(:, j))
%       trainStd(:, j) = (train(:, j) - mu_j) / sd_j
%       testStd (:, j) = (test (:, j) - mu_j) / sd_j     % uses TRAIN mu/sd
%
%   stats = struct('mu', muRow, 'sd', sdRow, 'constantCol', logicalRow)
%
%   Constant columns (sd == 0) are passed through with sd treated as 1
%   so the returned slice is all zeros; stats.constantCol(j) = true.
%
%   See PROPOSED_METHODOLOGY.md §5.2.

    arguments
        train double
        test  double
    end

    [~, p1] = size(train);
    [~, p2] = size(test);
    assert(p1 == p2, 'standardize:dimMismatch', ...
        'train has %d cols, test has %d', p1, p2);

    muRow = mean(train, 1, 'omitnan');
    sdRow = std(train, 0, 1, 'omitnan');

    constantCol = sdRow == 0 | ~isfinite(sdRow);
    sdRow(constantCol) = 1;

    trainStd = (train - muRow) ./ sdRow;
    testStd  = (test  - muRow) ./ sdRow;

    stats.mu          = muRow;
    stats.sd          = sdRow;
    stats.constantCol = constantCol;
end
