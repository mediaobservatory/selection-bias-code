function [ R_idx, M, N, names ] = gdelt(path,subset_size, reload)
    if reload == 1
        % Get data
        data = csvread(path,1,0);
         
        % Get names
        fd = fopen('data/source_map.csv');
        line  = fgets(fd);
        names = strsplit(line,',');
        ids  = str2double(line);
    end

    % Extract indices
    R_idx_data = [data(:,1), data(:,2)];
    % Keep subset
    R_idx_u = R_idx_data(1:subset_size,:);
    % Remove duplicates
    R_idx = unique(R_idx_u,'rows');
    % Find sizes for BPR
    extrema = max(R_idx);
    M = extrema(1) + 1;
    N = extrema(2) + 1;
end