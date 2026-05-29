function T = tableT3logml(bundle)
% tableT3logml  T3: log marginal likelihood per model (NaN for MLE GARCH).
%   Higher is better. Reads bundle.models only.
    arguments
        bundle (1,1) struct
    end
    M = bundle.models;
    model = {M.name}';
    logml = [M.logMarginalLik]';
    T = table(model, logml, 'VariableNames', {'Model', 'LogMarginalLik'});
end
