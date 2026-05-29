function fig = figForecastBands(bundle, opts)
% figForecastBands  F1: OOS returns with 95% one-step-ahead bands from the
%   proposed model. opts.file (stem) -> writes <stem>.png + <stem>.pdf.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    names = bundle.comparison.modelNames;
    ix    = find(strcmp(names, bundle.comparison.proposedName), 1);
    rec   = bundle.models(ix);
    yTest = bundle.data.y(bundle.data.testIdx);
    nu    = rec.nu;
    scale = sqrt((nu - 2) / nu) .* sqrt(rec.varForecast);
    band  = tinv(0.975, nu) .* scale;

    fig = figure('Visible', opts.Visible);
    t = (1:numel(yTest))';
    fill([t; flipud(t)], [band; flipud(-band)], [0.85 0.9 1.0], ...
        'EdgeColor', 'none'); hold on;
    plot(t, yTest, 'k.', 'MarkerSize', 6);
    plot(t, band, 'b-', t, -band, 'b-');
    xlabel('out-of-sample t'); ylabel('return');
    title(sprintf('F1: 95%% one-step bands (%s)', rec.name), 'Interpreter', 'none');
    hold off;
    report.saveFigure(fig, opts.file);
end
