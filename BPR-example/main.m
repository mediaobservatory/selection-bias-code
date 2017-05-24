% This code is a basic example of one-class Matrix Factorization
% using AUC as a ranking metric and Bayesian Personalized Ranking
% as an optimization procedure (https://arxiv.org/abs/1205.2618).
%clear;

% TODO 
% * Cross validation


iter     =   2e7; % number of iterations
alpha    =  0.05; % learning rate
lambda   =  0.01; % regularizer
sigma    =   0.1; % std for random initialization
mu       =   0.0; % mean for random initialization
K        =    20; % number of latent factors
reload   =     1; % Reload data
subset   =   1e6; % Don't load entire dataset
tetr_ratio = 0.2; % Test Train ratio
path     = 'data/hashed.csv'; % Path to dataset

% M events
% N sources
% R_idx is an nx2 matrix holding the indices of positive signals
% names holds the string representation of sources
[R_idx, M, N, names, ids] = gdelt(path, subset, reload);

%% Create testing and training sets

tetr_split = 3;


if tetr_split == 1 % Leave one out test-train split
    Rall = sparse(R_idx(:,1), R_idx(:,2), 1);
    idx_te = zeros(N,1); % Test indices

    % per source
    for i=1:N
       idxs = find(R_idx(:,1)==i);
       rand_idx = randi(length(idxs), 1);
       idx_te(i) = idxs(rand_idx);
    end
    
    % Create index mask
    % Test
    test_mask         = zeros(length(R_idx), 1);
    test_mask(idx_te) = 1;
    test_mask         = logical(test_mask);
    % Train
    train_mask = ~test_mask;

    R_idx_tr = R_idx(train_mask, :);
    R_idx_te = R_idx(test_mask , :);

    Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1, N, M);
    Rte  = sparse(R_idx_te(:,1), R_idx_te(:,2), 1, N, M);
    
elseif tetr_split == 2     % Random test-train split
    Rall = sparse(R_idx(:,1), R_idx(:,2), 1);
    datalen  = length(R_idx);
    rp       = randperm(datalen);
    pivot    = ceil(datalen/10);
    R_idx_te = R_idx(rp(1:pivot),:);
    R_idx_tr = R_idx(rp(pivot+1:end),:);

    % Create the User-Item interaction matrix
    Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1);
    
elseif tetr_split == 3   % Train = 1w ; Test = 1d
    [R_idx_te, M_te, N_te, ids_test , names_test ] = gdelt_weekly_te('data/hashed', reload, subset * tetr_ratio);
    [R_idx_tr, M_tr, N_tr, ids_train, names_train] = gdelt_weekly_tr('data/hashed', reload, subset * (1-tetr_ratio));
    
    M = max(M_te,M_tr);
    N = max(N_te,N_tr);
    
    R_idx = union(R_idx_te, R_idx_tr, 'rows');
    Rall = sparse(R_idx(:,1), R_idx(:,2), 1);
    idx_te = []; % Test indices
    
    % per source
    for i=1:N_te
       idxs = find(R_idx_te(:,1)==i);
       if length(idxs) > 0
        rand_idx = randi(length(idxs), 1);
        idx_te = [idx_te; idxs(rand_idx)];
       end
    end
    
    %%%%
    
    not_idx_te = zeros(length(R_idx_te),1);
    not_idx_te(idx_te) = logical(1);
    not_idx_te = ~not_idx_te;
    
    R_idx_tr = [R_idx_tr;R_idx_te(not_idx_te,:)];
    R_idx_te(not_idx_te,:) = [];
    %%%%
    
    
    Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1, N, M);
    Rte  = sparse(R_idx_te(:,1), R_idx_te(:,2), 1, N, M);
end

if length(R_idx) ~= nnz(Rall) & tetr_split ~= 3
    disp('Problem in Rall.')
elseif length(union(R_idx_te, R_idx_tr, 'rows')) ...
        ~= nnz(Rall) & tetr_split == 3
    disp('Problen in Rall (tetr==3)')
    disp(length(R_idx_tr)+length(R_idx_te) - nnz(Rall))
end

%% Run BPR

% Record auc values
auc_vals = zeros(iter/100000,1);

% Initialize low-rank matrices with random values
P = sigma.*randn(N,K) + mu; % Sources
Q = sigma.*randn(K,M) + mu; % Events

for step=1:iter
    
    % Select a random positive example
    i  = randi([1 length(R_idx_tr)]);
    iu = R_idx_tr(i,1);
    ii = R_idx_tr(i,2);
    
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
            te_iu = R_idx_te(i,1);
            te_ii = R_idx_te(i,2);
            te_ji = sample_neg(Rall, te_iu);
            
            sp = P(te_iu,:)*Q(:,te_ii);
            sn = P(te_iu,:)*Q(:,te_ji);
            
            if sp>sn; auc=auc+1; elseif sp==sn; auc=auc+0.5; end
        end
        auc = auc / length(R_idx_te);
        fprintf(['AUC test: ',num2str(auc),'\n']);
        auc_vals(step/100000) = auc;
    end
    
end

%% t-SNE plot for users' latent factors

addpath('tSNE_matlab/');
plot_top_20 = 1;      % Show locations of top 20 news_sources
plot_names  = 1;      % Plot names on scatter
plot_subset = 1:1000; % Only plot top 1K sources

% Get index of top 1K sources
[~,I]  = sort(sum(Rall, 2), 1, 'descend');
subidx = I(plot_subset);

% Run t-SNE on subset
ydata = tsne(P(subidx,:));

%%
if plot_top_20 == 1
    top_20_str = {'cnn.com', 'bbc.com', 'nytimes.com', 'foxnews.com', ...
                  'washingtonpost.com', 'washingtonpost.com', 'usatoday.com', ...
                  'theguardian.com', 'dailymail.co.uk', 'chinadaily.com.cn', ...
                  'telegraph.co.uk', 'wsj.com', 'indiatimes.com', 'independent.co.uk', ...
                  'elpais.com', 'lemonde.fr', 'ft.com', 'bostonglobe.com', ...
                  'ap.org', 'afp.com', 'reuters.com', 'yahoo.com', };
    top_20_ids = [];

    for ii=1:length(top_20_str)
        iid = find(strcmp(top_20_str{ii}, names_train));
        top_20_ids = [top_20_ids; iid];
    end
                    
    plot_idx = ismember(subidx,top_20_ids);
else
    plot_idx = subidx;
end

% Scatter plot t-SNE results
figure;
scatter(ydata(~plot_idx,1),ydata(~plot_idx,2));
hold on;
scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, 'r', 'filled');

% Overlay names
if plot_names == 1
    dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
    if tetr_split == 1
        c = names(subidx);
    elseif tetr_split == 3
        c = names_train(subidx);
    end
    text(ydata(:,1)+dx, ydata(:,2)+dy, c);
end
hold off


%% Plot Distance to Reuters + AP 

% Reuters
reuters_id  = find(strcmp('reuters.com', names_train));
reuters     = Rall(reuters_id,:);
reuters_idx = find(subidx == reuters_id);

% Associated Press
ap_id  = find(strcmp('ap.org', names_train));
ap     = Rall(ap_id,:);
ap_idx = find(subidx == ap_id);

% Compute distance
dist = @(id, source) nnz(source & Rall(id,:)) / sum(source);

recompute_dist = 1;

if recompute_dist == 1
    dist_reuters = zeros(1, length(subidx));
    dist_ap      = zeros(1, length(subidx));

    for i=1:length(subidx)
        source          = Rall(subidx(i),:);
        dist_reuters(i) = dist(reuters_id, source);
        dist_ap(i)      = dist(ap_id, source);
    end

end

% Plot
figure;
scatter(ydata(:,1), ydata(:,2), [], dist_ap);
hold on;

% Scatter
scatter(ydata(reuters_idx,1), ydata(reuters_idx,2), 300, 'r', 'filled');
scatter(ydata(ap_idx,1),      ydata(ap_idx,2),      300, 'r', 'filled');
% Overlay names
text(ydata(reuters_idx,1) + dx, ydata(reuters_idx,2) + dy, 'reuters');
text(ydata(ap_idx,1)      + dx, ydata(ap_idx,2)      + dy, 'ap');

figure;
scatter(ydata(:,1), ydata(:,2), [], dist_reuters);
hold on;

% Scatter
scatter(ydata(reuters_idx,1), ydata(reuters_idx,2), 300, 'r', 'filled');
scatter(ydata(ap_idx,1),      ydata(ap_idx,2),      300, 'r', 'filled');
% Overlay names
text(ydata(reuters_idx,1) + dx, ydata(reuters_idx,2) + dy, 'reuters');
text(ydata(ap_idx,1)      + dx, ydata(ap_idx,2)      + dy, 'ap');

%% DBSCAN

addpath('DBSCAN/')

epsilon=2;
MinPts=5;
X = ydata;
db=DBSCAN(X, epsilon, MinPts);
PlotClusterinResult(X, db);


%% Find recommendation ranking for holdout test event
% Manually curated top_20

for i=1:length(top_20_ids)
    search = top_20_ids(i);
    if search < N
        names_test(search)
        % dot product : P(i) . Q
        C = sum(bsxfun(@times, P(search,:), Q'), 2);
        % Bring down training indices
        tr_idx = find(Rtr(search,:));
        C(tr_idx) = -1000;
        % Sort recommendations
        [~,I_d] = sort(C, 1, 'descend');
        % Get the hold out event ID
        holdout_event = find(Rte(search,:));
        holdout_event_id = holdout_event(1);
        global_id = ids_test(holdout_event_id)+1
        % Find its ranking
        ranking = find(I_d==holdout_event_id)
    end
end

%% Recommendations for auto top_20

auto_top_20_ids = subidx(1:20);

for i=1:length(auto_top_20_ids)
    search = auto_top_20_ids(i);
    if search < N
        names_test(search)
        % dot product : P(i) . Q
        C = sum(bsxfun(@times, P(search,:), Q'), 2);
        % Bring down training indices
        tr_idx = find(Rtr(search,:));
        C(tr_idx) = -1000;
        % Sort recommendations
        [~,I_d] = sort(C, 1, 'descend');
        % Get the hold out event ID
        holdout_event = find(Rte(search,:));
        if numel(holdout_event) > 0
            holdout_event_id = holdout_event(1);
            global_id = ids_test(holdout_event_id)+1
            % Find its ranking
            ranking = find(I_d==holdout_event_id)
        else 
            'No holdout found'
        end
        
    end
end

%% Ranking Jay

auto_top_20_ids = subidx(end-1000:end);
unique_te = unique(R_idx_te(:,2));

res = [];
for i=1:length(R_idx_te)
   te_ev = R_idx_te(i,:);
   
   sp    = P(te_ev(1),:)*Q(:,te_ev(2));
   if any(te_ev(1)==auto_top_20_ids)
   cnt = 1;
   for j=1:length(unique_te)
     if i==j;continue;end
     sn = P(te_ev(1),:)*Q(:,R_idx_te(j,2));
   
     if sn>sp;cnt=cnt+1;end;
   end
   res = [res;cnt];
   %disp(['rank: ',num2str(cnt)]);
   end
end

%% Sanity Jay

unique_te = unique(R_idx_te(:,2));

auc = 0;
for i=1:length(R_idx_te)
   te_ev = R_idx_te(i,:);
   sp = P(te_ev(1),:)*Q(:,te_ev(2));
   
   te_i = te_ev(2);
   while te_i == te_ev(2)
     rand_i  = randi([1 length(R_idx_te)]);
     te_i = R_idx_te(rand_i,2);
   end
   
   sn = P(te_ev(1),:)*Q(:,te_i);
   if sp>sn; auc=auc+1; elseif sp==sn; auc=auc+0.5; end

end

auc = auc / length(R_idx_te);
fprintf(['AUC test: ',num2str(auc),'\n']);


