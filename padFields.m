function flatStruct = padFields(flatStruct, padValue)
    if nargin < 2
        padValue = NaN; % Default padding value
    end

    % Get the maximum number of rows across all fields
    fieldNames = fieldnames(flatStruct);
    maxRows = max(cellfun(@(f) size(flatStruct.(f), 1), fieldNames)); % Get row sizes

    % Pad each field to the maximum number of rows
    for i = 1:numel(fieldNames)
        field = fieldNames{i};
        value = flatStruct.(field);

        % Check if the field is numeric and multi-column
        if isnumeric(value) && ismatrix(value)
            % Get current number of rows
            currentRows = size(value, 1);

            % Pad rows if needed
            if currentRows < maxRows
                padding = repmat(padValue, maxRows - currentRows, size(value, 2));
                flatStruct.(field) = [value; padding];
            end
        else
            error('Unsupported data type or unexpected structure in field "%s".', field);
        end
    end
end