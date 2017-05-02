% This code is a basic example of one-class Matrix Factorization
% using AUC as a ranking metric and Bayesian Personalized Ranking
% as an optimization procedure (https://arxiv.org/abs/1205.2618).
%clear;

iter     =   2e6; % number of iterations
alpha    =  0.05; % learning rate
lambda   =  0.01; % regularizer
sigma    =   0.1; % std for random initialization
mu       =   0.0; % mean for random initialization
K        =    20; % number of latent factors

% Create synthetic dataset
% simulate a one-class collaborative filtering setup
% => positive signals only
reload = 0;
subset_size = 1e6;
path = 'data/source_map.csv';

[R_idx, M, N] = gdelt(path, subset_size, reload);


%%

% Split test-train
datalen  = size(R_idx,1);
rp       = randperm(datalen);
pivot    = ceil(datalen/10);
R_idx_te = R_idx(rp(1:pivot),:);
R_idx_tr = R_idx(rp(pivot+1:end),:);

% Create the User-Item interaction matrix
Rall = sparse(R_idx(:,2), R_idx(:,1), 1);
Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1); 

if length(R_idx) ~= length(nonzeros(Rall))
    disp('Problem in Rall.')
end

%%
% Initialize low-rank matrices with random values
P = sigma.*randn(N,K) + mu; % Events
Q = sigma.*randn(K,M) + mu; % Sources

for step=1:iter
    
    % Select a random positive example
    i  = randi([1 length(R_idx_tr)]);
    iu = R_idx_tr(i,2);
    ii = R_idx_tr(i,1);
    
    % Sample a negative example
    ji = sample_neg(Rtr,iu);
    
    % See BPR paper for details
    px = (P(iu,:) * (Q(:,ii)-Q(:,ji)));
    z = 1 /(1 + exp(px));
    
    % update P
    d = (Q(:,ii)-Q(:,ji))*z - lambda*P(iu,:)';
    P(iu,:) = P(iu,:) + alpha*d';
    
    % update Q positive
    d = P(iu,:)*z - lambda*Q(:,ii)';
    Q(:,ii) = Q(:,ii) + alpha*d';
    
    % update Q negative
    d = -P(iu,:)*z - lambda*Q(:,ji)';
    Q(:,ji) = Q(:,ji) + alpha*d';
    
    if mod(step,100000)==0
        
        % Compute the Area Under the Curve (AUC)
        auc = 0;
        for i=1:length(R_idx_te)
            te_i  = randi([1 length(R_idx_te)]);
            te_iu = R_idx_te(i,2);
            te_ii = R_idx_te(i,1);
            te_ji = sample_neg(Rall,te_iu);
            
            sp = P(te_iu,:)*Q(:,te_ii);
            sn = P(te_iu,:)*Q(:,te_ji);
            
            if sp>sn; auc=auc+1; elseif sp==sn; auc=auc+0.5; end
        end
        auc = auc / length(R_idx_te);
        fprintf(['AUC test: ',num2str(auc),'\n']);
    end
    
end

%% t-SNE plot for users' latent factors
addpath('tSNE_matlab/');
plot_top_20 = 1;
plot_names = 1;


[~,I] = sort(sum_source, 1,'descend');
subidx = I(1:1000);
ydata = tsne(P(subidx,:));

if plot_top_20 == 1
                % cnn, bbc, nyt, fox, wapo, usat, gua, dma, chd, tlg, wsj, IT
    top_20_ids = [186, 612, 367, 211, 328 , 1725, 611, 502, 160, 614, 865, 92, ...
                  ... % indp, pais, lmnde, FT,   BG  , AP,  AFP, reu
                        1388, 7537, 19048, 8030, 397, 161, 4236, 297];
                    
    plot_idx = ismember(subidx,top_20_ids);
else
    plot_idx = subidx;
end

figure;
scatter(ydata(~plot_idx,1),ydata(~plot_idx,2));
hold on;
scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, 'r', 'filled');

if plot_names == 1
    dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
    c = names(subidx);
    text(ydata(:,1)+dx, ydata(:,2)+dy, c);
end
hold off


%% Plot Distance to Reuters + AP 

% Reuters
reuters_id = find(strcmp('reuters.com', names));
reuters = Rall(reuters_id,:);
reuters_idx = find(subidx == reuters_id);

% Associated Press
ap_id = find(strcmp('ap.org', names));
ap = Rall(ap_id,:);
ap_idx = find(subidx == ap_id);

% Compute distance
dist = @(id, source) nnz(source & Rall(id,:)) / sum(source);

recompute_dist = 1;

if recompute_dist == 1
    dist_reuters = [];
    dist_ap = [];

    for i=1:length(subidx)
        i
        source = Rall(subidx(i),:);
        dist_reuters = [dist_reuters; dist(reuters_id, source)];
        dist_ap      = [dist_ap; dist(ap_id, source)];
    end

end


figure;
scatter(ydata(:,1),ydata(:,2),[],dist_ap);
hold on;

% Scatter
scatter(ydata(reuters_idx,1), ydata(reuters_idx,2), 300, 'r', 'filled');
scatter(ydata(ap_idx,1),      ydata(ap_idx,2),      300, 'r', 'filled');
% Overlay names
text(ydata(reuters_idx,1)+dx, ydata(reuters_idx,2)+dy, 'reuters');
text(ydata(ap_idx,1)+dx,      ydata(ap_idx,2)+dy,      'ap');

