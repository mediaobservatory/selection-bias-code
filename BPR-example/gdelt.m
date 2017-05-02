function [ R_idx, M, N ] = gdelt(path,subset_size, reload)
    if reload == 1
        fd = fopen(path);

        line  = fgets(fd);
        names = strsplit(line,',');

        line = fgets(fd);
        line = strsplit(line,',');

        ids  = str2double(line);
        path = 'data/hashed.csv';
        data = csvread(path,1,0);
    end


    R_idx_data = [data(:,1), data(:,2)];
    R_idx_u = R_idx_data(1:subset_size,:);
    R_idx = unique(R_idx_u,'rows');
    extrema = max(R_idx);
    M = extrema(1) + 1;
    N = extrema(2) + 1;
end