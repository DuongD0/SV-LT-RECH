function opts = withDefaults(opts)
% withDefaults  Shared figure-option defaults for the report.fig* builders.
    if ~isfield(opts, 'Visible'); opts.Visible = 'off'; end
    if ~isfield(opts, 'file');    opts.file    = '';    end
end
