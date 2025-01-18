function flatStruct = flattenStruct(nestedStruct, parentName, delimiter)
    if nargin < 2
        parentName = '';
    end
    if nargin < 3
        delimiter = '_'; % Default delimiter
    end

    flatStruct = struct(); % Initialize an empty flat struct
    fields = fieldnames(nestedStruct);

    for i = 1:numel(fields)
        fieldName = fields{i};
        fullName = fieldName;
        if ~isempty(parentName)
            fullName = [parentName delimiter fieldName]; % Concatenate using delimiter
        end

        value = nestedStruct.(fieldName);

        if isstruct(value)
            % Recursive call for nested structs
            nestedFlat = flattenStruct(value, fullName, delimiter);

            % Merge nestedFlat into the current flatStruct
            flatStruct = mergeStructs(flatStruct, nestedFlat);
        else
            % Add non-struct values
            flatStruct.(fullName) = value;
        end
    end
end

% Improved mergeStructs Function
function merged = mergeStructs(struct1, struct2)
    % Combine field names
    fields1 = fieldnames(struct1);
    fields2 = fieldnames(struct2);
    allFields = unique([fields1; fields2]);

    % Initialize merged struct
    merged = struct();

    % Populate merged struct with values from struct1 and struct2
    for i = 1:numel(allFields)
        field = allFields{i};

        if isfield(struct1, field)
            merged.(field) = struct1.(field);
        else
            merged.(field) = [];
        end

        if isfield(struct2, field)
            merged.(field) = struct2.(field);
        end
    end
end



    