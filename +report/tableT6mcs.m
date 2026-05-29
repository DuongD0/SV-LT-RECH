function T = tableT6mcs(bundle)
% tableT6mcs  T6: Model Confidence Set membership + MCS p-value per model.
%   Reads bundle.comparison.{modelNames,mcs}.
    arguments
        bundle (1,1) struct
    end
    names = bundle.comparison.modelNames(:);
    mcs   = bundle.comparison.mcs;
    T = table(names, logical(mcs.inSet(:)), mcs.pValues(:), ...
        'VariableNames', {'Model','inMCS','pValue'});
end
