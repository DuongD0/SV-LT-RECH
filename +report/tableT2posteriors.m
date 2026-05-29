function T = tableT2posteriors(bundle)
% tableT2posteriors  T2: posterior mean + std of every parameter per model.
%   Long format (heterogeneous parameter sets across models). GARCH point
%   estimates carry NaN std. Reads bundle.models only.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    modelCol = {}; paramCol = {}; meanCol = []; stdCol = [];
    for m = 1:numel(M)
        names = M(m).paramNames;
        mu    = M(m).posteriorMean;
        sd    = M(m).posteriorStd;
        for p = 1:numel(names)
            modelCol{end+1, 1} = M(m).name;       %#ok<AGROW>
            paramCol{end+1, 1} = names{p};        %#ok<AGROW>
            meanCol(end+1, 1)  = mu(p);           %#ok<AGROW>
            if isempty(sd)
                stdCol(end+1, 1) = NaN;           %#ok<AGROW>
            else
                stdCol(end+1, 1) = sd(p);         %#ok<AGROW>
            end
        end
    end
    T = table(modelCol, paramCol, meanCol, stdCol, ...
        'VariableNames', {'Model','Parameter','PosteriorMean','PosteriorStd'});
end
