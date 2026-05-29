function fig = figCovariatePosterior(bundle, opts)
% figCovariatePosterior  F3: kernel-density posteriors of the covariate
%   coefficients v_z for the proposed model. opts.file -> PNG + PDF.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    names = bundle.comparison.modelNames;
    ix    = find(strcmp(names, bundle.comparison.proposedName), 1);
    rec   = bundle.models(ix);
    cols  = find(startsWith(rec.paramNames, 'v_z_'));

    fig = figure('Visible', opts.Visible);
    hold on;
    leg = {};
    for c = cols
        [f, xi] = ksdensity(rec.theta(:, c));
        plot(xi, f, 'LineWidth', 1.2);
        leg{end+1} = rec.paramNames{c}; %#ok<AGROW>
    end
    xline(0, 'k--');
    if ~isempty(leg); legend(leg, 'Interpreter', 'none', 'Location', 'best'); end
    xlabel('coefficient value'); ylabel('posterior density');
    title(sprintf('F3: covariate-coefficient posteriors (%s)', rec.name), ...
        'Interpreter', 'none');
    hold off;
    report.saveFigure(fig, opts.file);
end
