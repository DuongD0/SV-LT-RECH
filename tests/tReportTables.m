classdef tReportTables < matlab.unittest.TestCase
% tReportTables  Report table builders + LaTeX renderer over a synthetic bundle.

    properties
        bundle
    end

    methods (TestClassSetup)
        function loadFixture(testCase)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, 'fixtures'));
            testCase.bundle = syntheticBundle();
        end
    end

    methods (Test)

        function latexBoldsBestPerColumn(testCase)
            T = table({'A';'B';'C'}, [1.0;0.5;0.8], [2.0;3.0;1.0], ...
                'VariableNames', {'model','QLIKE','LogML'});
            tex = report.latexTable(T, struct( ...
                'lowerBetter', {{'QLIKE'}}, 'higherBetter', {{'LogML'}}, ...
                'caption', 'Demo', 'label', 'tab:demo'));
            testCase.verifyClass(tex, 'char');
            testCase.verifySubstring(tex, '\begin{tabular}');
            testCase.verifySubstring(tex, '\textbf{0.5');   % min QLIKE bolded
            testCase.verifySubstring(tex, '\textbf{3');      % max LogML bolded
            testCase.verifySubstring(tex, '\bottomrule');
        end

        function latexRendersNaNAsDash(testCase)
            T = table({'A'}, NaN, 'VariableNames', {'model','QLIKE'});
            tex = report.latexTable(T, struct('lowerBetter', {{'QLIKE'}}));
            testCase.verifySubstring(tex, '--');
        end

        function t1HasDescriptivesAndDiagnostics(testCase)
            T = report.tableT1descriptives(testCase.bundle);
            testCase.verifyClass(T, 'table');
            stats = T.Statistic;
            testCase.verifyTrue(any(strcmp(stats, 'Kurtosis')));
            testCase.verifyTrue(any(strcmp(stats, 'ADF p-value')));
            testCase.verifyTrue(any(strcmp(stats, 'BDS stat')));
            testCase.verifyTrue(all(isfinite(T.Value)));
        end

        function t2OneRowPerParameterPerModel(testCase)
            T = report.tableT2posteriors(testCase.bundle);
            testCase.verifyClass(T, 'table');
            testCase.verifyTrue(all(ismember( ...
                {'Model','Parameter','PosteriorMean','PosteriorStd'}, ...
                T.Properties.VariableNames)));
            srnRows = T(strcmp(T.Model, 'SVLTRECH-SRN'), :);
            testCase.verifyTrue(any(strcmp(srnRows.Parameter, 'v_z_1')));
        end

        function t3ListsLogMLPerSvModel(testCase)
            T = report.tableT3logml(testCase.bundle);
            testCase.verifyTrue(all(ismember({'Model','LogMarginalLik'}, ...
                T.Properties.VariableNames)));
            srn = T(strcmp(T.Model, 'SVLTRECH-SRN'), :);
            testCase.verifyTrue(isfinite(srn.LogMarginalLik));
        end

        function t4ScoresOneRowPerModel(testCase)
            T = report.tableT4scores(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember( ...
                {'Model','PPS','QS1','QS5','MSE','MAE','R2LOG','QLIKE'}, ...
                T.Properties.VariableNames)));
        end

        function t5DmRowPerBaseline(testCase)
            T = report.tableT5dm(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models) - 1);
            testCase.verifyTrue(all(ismember({'Baseline','DMstat','pValue'}, ...
                T.Properties.VariableNames)));
        end

        function t6McsMembership(testCase)
            T = report.tableT6mcs(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember({'Model','inMCS','pValue'}, ...
                T.Properties.VariableNames)));
            testCase.verifyTrue(islogical(T.inMCS) || isnumeric(T.inMCS));
        end

        function t7ResidualMomentsPerModel(testCase)
            T = report.tableT7residuals(testCase.bundle);
            testCase.verifyEqual(height(T), numel(testCase.bundle.models));
            testCase.verifyTrue(all(ismember( ...
                {'Model','Mean','Std','Skewness','Kurtosis','LBQ2_pValue'}, ...
                T.Properties.VariableNames)));
            testCase.verifyTrue(all(isfinite(T.Std)));
        end

        function orchestratorEmitsAllArtifacts(testCase)
            addpath('scripts');
            tmp = tempname; mkdir(tmp);
            testCase.addTeardown(@() rmdir(tmp, 's'));
            bundle = testCase.bundle;                       %#ok<NASGU>
            save(fullfile(tmp, 'bundle.mat'), 'bundle');

            make_paper_tables(tmp);

            for t = 1:7
                csv = fullfile(tmp, 'tables', sprintf('T%d.csv', t));
                tex = fullfile(tmp, 'tables', sprintf('T%d.tex', t));
                testCase.verifyTrue(isfile(csv), sprintf('missing %s', csv));
                testCase.verifyTrue(isfile(tex), sprintf('missing %s', tex));
            end
            for f = 1:4
                png = fullfile(tmp, 'figures', sprintf('F%d.png', f));
                testCase.verifyTrue(isfile(png), sprintf('missing %s', png));
            end
        end
    end
end
