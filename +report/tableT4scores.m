function T = tableT4scores(bundle)
% tableT4scores  T4: the five predictive scores + QLIKE, one row per model.
%   Reads bundle.comparison.{modelNames,scores}.
    arguments
        bundle (1,1) struct
    end
    names = bundle.comparison.modelNames(:);
    S     = bundle.comparison.scores;
    T = table(names, [S.pps]', [S.qs1]', [S.qs5]', [S.mse]', [S.mae]', ...
        [S.r2log]', [S.qlike]', ...
        'VariableNames', {'Model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'});
end
