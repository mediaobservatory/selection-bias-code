function [ R_idx_tr, M_tr, N_tr, ids_train, names_train ] = gdelt_weekly_tr(base_path, week, reload, subset_size)
    if reload == 1
        
        % Get data
        train_path = strcat(base_path, '_', week, '_train.csv');
        train_data = csvread(train_path,1,0);
         
        % Get names
        % Test
        fd   = fopen(strcat('data/source_map_',week,'.csv'));
        line = fgets(fd);
        
        names_train = strsplit(line, ',');
        
        fd       = fopen(strcat('data/event_map_',week,'.csv'));
        line     = fgets(fd);
        line     = strsplit(line,',');
        ids_train = str2double(line);
    end

    % Extract indices
    R_idx_data = [train_data(:,2), train_data(:,1)];
    % Remove duplicates
    R_idx_tr = unique(R_idx_data,'rows');
    R_idx_tr = datasample(R_idx_tr,subset_size);
    % Find sizes for BPR
    extrema = max(R_idx_tr);
    M_tr = extrema(2) ; % Events
    N_tr = extrema(1) ; % Sources
end