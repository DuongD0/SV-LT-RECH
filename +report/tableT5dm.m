function T = tableT5dm(bundle)
% tableT5dm  T5: Diebold-Mariano statistic + p-value, each baseline vs the
%   proposed model. Negative stat => baseline has higher loss (proposed wins).
%   Reads bundle.comparison.{dmBaselines,dmVsProposed}.
    arguments
        bundle (1,1) struct
    end
    base = bundle.comparison.dmBaselines(:);
    dm   = bundle.comparison.dmVsProposed;
    T = table(base, [dm.statistic]', [dm.pValue]', ...
        'VariableNames', {'Baseline','DMstat','pValue'});
end
