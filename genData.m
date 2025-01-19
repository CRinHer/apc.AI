function [curves, nums, time] = genData(list)
%list = number of systems you want to generate

%Random relevancy numbers to generate
nums = [];

%set time equal to empty struct
time = struct();
%set curves equal to empty struct
curves = struct();

%for every system you want to generate
for i = 1:list

    %Run solve random model to generate random systems and run a step test
    [y, t] = solveRandomModel;
    fieldName = sprintf('System_%d',i);
    curves.(fieldName) = y;

    numsLen = length(curves.(fieldName));

    nums = [nums rand(64,1)];

    newField = sprintf('time_%d',i);

    %All the graphs in every system is based on the same time. So each
    %vector of times t is related to the same numbered graph. System_1 is
    %associated with time_1
    time.(newField) = t;


end

end
