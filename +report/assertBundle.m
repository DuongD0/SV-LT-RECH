function assertBundle(bundle)
% assertBundle  Validate the top-level shape of a results bundle.
%   Throws a clear error if a required field is missing. Called once by
%   scripts/make_paper_tables before any builder runs.
    arguments
        bundle (1,1) struct
    end
    req = {'meta','data','descriptives','models','comparison'};
    for k = 1:numel(req)
        assert(isfield(bundle, req{k}), 'report:assertBundle:missingField', ...
            'bundle is missing required field "%s"', req{k});
    end
    assert(~isempty(bundle.models), 'report:assertBundle:noModels', ...
        'bundle.models is empty');
    cmpReq = {'modelNames','scores','qlAll','dmVsProposed','mcs','proposedName'};
    for k = 1:numel(cmpReq)
        assert(isfield(bundle.comparison, cmpReq{k}), ...
            'report:assertBundle:missingComparisonField', ...
            'bundle.comparison is missing "%s"', cmpReq{k});
    end
end
