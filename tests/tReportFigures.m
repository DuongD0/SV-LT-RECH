classdef tReportFigures < matlab.unittest.TestCase
% tReportFigures  F1-F4 builders emit non-empty PNG + PDF from a bundle.

    properties
        bundle
        tmp
    end

    methods (TestClassSetup)
        function setup(testCase)
            here = fileparts(mfilename('fullpath'));
            addpath(fullfile(here, 'fixtures'));
            testCase.bundle = syntheticBundle();
            testCase.tmp = tempname; mkdir(testCase.tmp);
            testCase.addTeardown(@() rmdir(testCase.tmp, 's'));
        end
    end

    methods (Test)
        function f1ForecastBands(testCase)
            f = fullfile(testCase.tmp, 'F1');
            fig = report.figForecastBands(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end

        function f2QQ(testCase)
            f = fullfile(testCase.tmp, 'F2');
            fig = report.figQQ(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end

        function f3CovariatePosterior(testCase)
            f = fullfile(testCase.tmp, 'F3');
            fig = report.figCovariatePosterior(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end

        function f4OmegaState(testCase)
            f = fullfile(testCase.tmp, 'F4');
            fig = report.figOmegaState(testCase.bundle, struct('file', f));
            testCase.verifyTrue(isgraphics(fig));
            close(fig);
            assertNonEmptyFiles(testCase, f);
        end
    end
end

function assertNonEmptyFiles(testCase, stem)
    for ext = ["png", "pdf"]
        p = stem + "." + ext;
        d = dir(p);
        testCase.verifyNotEmpty(d, sprintf('missing %s', p));
        testCase.verifyGreaterThan(d.bytes, 0);
    end
end
