function make_paper_tables(resultDir)
% make_paper_tables  Build paper tables T1-T7 + figures F1-F4 from a bundle.
%
%   make_paper_tables(resultDir)
%
%   Loads <resultDir>/bundle.mat and writes:
%     <resultDir>/tables/T1..T7.{csv,tex}
%     <resultDir>/figures/F1..F4.{png,pdf}
%
%   The report layer is a pure consumer of the bundle (design spec §2/A);
%   no inference is re-run. See PROPOSED_METHODOLOGY.md §12.

    arguments
        resultDir (1,:) char
    end

    S = load(fullfile(resultDir, 'bundle.mat'), 'bundle');
    bundle = S.bundle;
    report.assertBundle(bundle);

    tDir = fullfile(resultDir, 'tables');
    fDir = fullfile(resultDir, 'figures');
    if ~isfolder(tDir); mkdir(tDir); end
    if ~isfolder(fDir); mkdir(fDir); end

    %% Tables (CSV + LaTeX). Columns: number, builder, lower-better, higher-better.
    specs = {
        1, @report.tableT1descriptives, {},                                  {}
        2, @report.tableT2posteriors,   {},                                  {}
        3, @report.tableT3logml,        {},                                  {'LogMarginalLik'}
        4, @report.tableT4scores,       {'PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'}, {}
        5, @report.tableT5dm,           {},                                  {}
        6, @report.tableT6mcs,          {},                                  {}
        7, @report.tableT7residuals,    {},                                  {}
    };
    for i = 1:size(specs, 1)
        n   = specs{i, 1};
        T   = specs{i, 2}(bundle);
        csv = fullfile(tDir, sprintf('T%d.csv', n));
        tex = fullfile(tDir, sprintf('T%d.tex', n));
        writetable(T, csv);
        report.latexTable(T, struct('lowerBetter', specs(i,3), ...
            'higherBetter', specs(i,4), ...
            'caption', sprintf('Table T%d', n), 'label', sprintf('tab:T%d', n), ...
            'file', tex));
        fprintf('[make_paper_tables] wrote %s + .tex\n', csv);
    end

    %% Figures (PNG + PDF).
    figs = {1, @report.figForecastBands; 2, @report.figQQ; ...
            3, @report.figCovariatePosterior; 4, @report.figOmegaState};
    for i = 1:size(figs, 1)
        n   = figs{i, 1};
        fig = figs{i, 2}(bundle, struct('file', fullfile(fDir, sprintf('F%d', n))));
        close(fig);
        fprintf('[make_paper_tables] wrote F%d.{png,pdf}\n', n);
    end
end
