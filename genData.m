function [curves, nums] = genData(list)
%list = number of graphs you want to generate
nums = []
%set curves equal to empty struct
curves = struct();
for i = 1:list

    y = solveRandomModel;
    fieldName = sprintf('Graph_%d',i);
    curves.(fieldName) = y;

    numsLen = length(curves.(fieldName));

    nums = [nums rand(64,1)];


end


end
