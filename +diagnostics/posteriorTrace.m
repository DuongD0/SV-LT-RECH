function fig = posteriorTrace(result, names, opts)
% posteriorTrace  Plot SMC posterior particle distribution + anneal schedule.
%
%   fig = posteriorTrace(result, names)
%   fig = posteriorTrace(result, names, opts)
%
%   Renders one panel per parameter showing the posterior particle
%   histogram with an optional truth marker, plus a final panel for the
%   anneal-temperature schedule and ESS trace.
%
%   Inputs
%   ------
%     result : struct returned by inference.smc.likelihoodAnneal
%              (fields: theta, schedule, essTrace).
%     names  : 1-by-K cellstr of parameter names.
%     opts   : struct, optional
%                .thetaTrue (1-by-K) true theta to mark on each panel
%                .nBins     histogram bin count (default 40)
%                .title     figure title (default 'Posterior trace')
%                .Visible   'on' | 'off' (default 'off')

    arguments
        result struct
        names  cell
        opts   struct = struct()
    end

    defaults.thetaTrue = [];
    defaults.nBins     = 40;
    defaults.title     = 'Posterior trace';
    defaults.Visible   = 'off';
    opts = mergeStruct(defaults, opts);

    K     = numel(names);
    fig   = figure('Visible', opts.Visible);
    nCols = min(4, K + 1);
    nRows = ceil((K + 1) / nCols);

    for k = 1:K
        subplot(nRows, nCols, k);
        histogram(result.theta(:, k), opts.nBins);
        title(names{k}, 'Interpreter', 'none');
        if ~isempty(opts.thetaTrue)
            yLim = ylim;
            line([opts.thetaTrue(k), opts.thetaTrue(k)], yLim, ...
                 'Color', 'r', 'LineWidth', 1.5);
        end
    end

    subplot(nRows, nCols, K + 1);
    yyaxis left
    plot(result.schedule, 'LineWidth', 1);
    ylabel('a_k');
    yyaxis right
    plot(result.essTrace, 'LineWidth', 1);
    ylabel('ESS');
    title('Schedule / ESS');

    sgtitle(opts.title, 'Interpreter', 'none');
end


function out = mergeStruct(defaults, in)
    out = defaults;
    f = fieldnames(in);
    for k = 1:numel(f)
        out.(f{k}) = in.(f{k});
    end
end
