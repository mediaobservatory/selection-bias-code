function [ neg_i ] = sample_neg( R,u )
%SAMPLE_NEG sample an item that had no interaction with the given user
while true
    item = randi([1 size(R,2)]);
    if R(u,item) == 0; neg_i=item; break; end;
end
end

