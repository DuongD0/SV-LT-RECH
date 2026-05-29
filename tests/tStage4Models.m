classdef tStage4Models < matlab.unittest.TestCase
% tStage4Models  The 6-model Stage-4 registry has the expected shape.

    methods (Test)
        function registryShape(testCase)
            reg = experiments.stage4Models();
            names = {reg.name};
            testCase.verifyEqual(names, ...
                {'GARCH-t','GJR-t','SVLT','SVLTRECH-SRN', ...
                 'SVLTRECH-LSTM','SVLTRECH-GRU'});
            testCase.verifyEqual(sum([reg.isProposed]), 1);   % exactly one proposed
            testCase.verifyEqual(reg(strcmp(names,'SVLTRECH-SRN')).isProposed, true);
        end

        function svConstructorsBuild(testCase)
            reg = experiments.stage4Models();
            K   = 2;
            for i = 1:numel(reg)
                if strcmp(reg(i).kind, 'sv')
                    mdl = reg(i).ctor(K, 'cholesky');
                    testCase.verifyTrue(isa(mdl, 'models.Model'));
                end
            end
        end

        function garchEntriesCarryType(testCase)
            reg = experiments.stage4Models();
            g   = reg(strcmp({reg.name}, 'GJR-t'));
            testCase.verifyEqual(g.kind, 'garch');
            testCase.verifyEqual(g.garchType, 'gjr');
        end
    end
end
