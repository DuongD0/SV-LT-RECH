function fig = figOmegaState(bundle, opts)
% figOmegaState  F4: recurrent-state omega_t path for models that expose one
%   (the SRN flagship). Interpretability plot. opts.file -> PNG + PDF.
    arguments
        bundle (1,1) struct
        opts   struct = struct()
    end
    opts = report.withDefaults(opts);

    M   = bundle.models;
    fig = figure('Visible', opts.Visible);
    hold on;
    leg = {};
    for m = 1:numel(M)
        w = M(m).omegaPath;
        if isempty(w); continue; end
        plot(1:numel(w), w, 'LineWidth', 1.0);
        leg{end+1} = M(m).name; %#ok<AGROW>
    end
    if isempty(leg)
        text(0.5, 0.5, 'no recurrent-state model', 'HorizontalAlignment', 'center');
    else
        legend(leg, 'Interpreter', 'none', 'Location', 'best');
    end
    xlabel('in-sample t'); ylabel('\omega_t');
    title('F4: recurrent-state path \omega_t');
    hold off;
    report.saveFigure(fig, opts.file);
end
