function [ R_idx ] = gen_data(M,N)

num_pop =     5; % Number of population type
std     =  0.05;
R_idx   =    [];
for u=1:N
    pop         = randi([2 num_pop+1]);
    pop_bias    = ceil(M/(num_pop+2))*pop;
    pop_scale   = ceil(M/(num_pop+2))*std;
    num_signals = randi([10 20]);
    
    signals     = round(pop_scale*randn(num_signals,1)+pop_bias);
    user_col    = repmat(u,[length(signals) 1]);
    R_idx       = [R_idx;horzcat(user_col,signals)];
end


end

