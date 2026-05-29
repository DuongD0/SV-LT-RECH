function fig = figQQ(bundle, opts)
% figQQ  F2: QQ-plots of standardized residuals vs N(0,1), one panel per model.
%   opts.file (stem) -> writes <stem>.png + <stem>.pdf.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    M     = bundle.models;
    nM    = numel(M);
    nCols = min(3, nM);
    nRows = ceil(nM / nCols);

    fig = figure('Visible', opts.Visible);
    for m = 1:nM
        subplot(nRows, nCols, m);
        e = M(m).stdResid(:); e = e(isfinite(e));
        qqplot(e);
        title(M(m).name, 'Interpreter', 'none');
    end
    sgtitle('F2: standardized-residual QQ vs N(0,1)');
    report.saveFigure(fig, opts.file);
end
