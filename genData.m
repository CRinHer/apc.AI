function curves = genData(list)
%list = number of graphs you want to generate

%set curves equal to empty struct
curves = struct();
for i = 1:list

    y = solveRandomModel;
    fieldName = sprintf('Graph_%d',i);
    curves.(fieldName) = y;
end

end