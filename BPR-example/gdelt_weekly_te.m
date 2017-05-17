function [ R_idx_te, M_te, N_te, ids_test, names_test ] = gdelt_weekly_te(base_path, reload)
    if reload == 1
        
        % Get data
        test_path = strcat(base_path, '_1_day.csv');
        test_data = csvread(test_path,1,0);
         
        % Get names
        % Test
        fd   = fopen('data/source_map_1_day.csv');
        line = fgets(fd);
        
        names_test = strsplit(line, ',');
        
        fd       = fopen('data/event_map_1_day.csv');
        line     = fgets(fd);
        line     = strsplit(line,',');
        ids_test = str2double(line);
    end

    % Extract indices
    R_idx_data = [test_data(:,2), test_data(:,1)];
    % Remove duplicates
    R_idx_te = unique(R_idx_data,'rows');
    % Find sizes for BPR
    extrema = max(R_idx_te);
    M_te = extrema(2) ; % Events
    N_te = extrema(1) ; % Sources
end