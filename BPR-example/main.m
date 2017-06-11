% This code is a basic example of one-class Matrix Factorization
% using AUC as a ranking metric and Bayesian Personalized Ranking
% as an optimization procedure (https://arxiv.org/abs/1205.2618).
%clear;

% TODO 
% * Cross validation


iter     =   2e7; % number of iterations
alpha    =   0.1; % learning rate % TODO CV
lambda   =  0.01; % regularizer
sigma    =   0.1; % std for random initialization
mu       =   0.0; % mean for random initialization
K        =    20; % number of latent factors % TODO CV
reload   =     1; % Reload data
subset   =   1e6; % Don't load entire dataset
tetr_ratio = 0.2; % Test Train ratio
path     = 'data/hashed.csv'; % Path to dataset
%%
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

    % Create the Source-Event interaction matrix
    Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1);
    
elseif tetr_split == 3   % Train = 1w ; Test = 1d
    [R_idx_te, M_te, N_te, ids_test , names_test ] = gdelt_weekly_te('data/hashed', reload, subset * tetr_ratio);
    [R_idx_tr, M_tr, N_tr, ids_train, names_train] = gdelt_weekly_tr('data/hashed', reload, subset * (1-tetr_ratio));
    
    names = names_train;
    
    M = max(M_te,M_tr);
    N = max(N_te,N_tr);
    
    R_idx = union(R_idx_te, R_idx_tr, 'rows');
    Rall = sparse(R_idx(:,1), R_idx(:,2), 1);
    idx_te = []; % Test indices
    
    % Leave one out per source
    for i=1:N_te
       idxs = find(R_idx_te(:,1)==i);      % Find indices corresponding to source
       if ~isempty(idxs)
        rand_idx = randi(length(idxs), 1); % Randomly chose one
        idx_te = [idx_te; idxs(rand_idx)]; % Add it to the test indices
       end
    end

    % Keep only the heldout test samples
    % Create a mask
    not_idx_te = zeros(length(R_idx_te),1);
    not_idx_te(idx_te) = true; 
    not_idx_te = ~not_idx_te;
    
    R_idx_tr = [R_idx_tr;R_idx_te(not_idx_te,:)]; % Add the non-heldout events to the training set
    R_idx_te(not_idx_te,:) = [];                  % Remove them from the test set, leaving only the
                                                  % heldout samples
        
    % Create the Source-Event interaction matrix
    Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1, N, M);
    Rte  = sparse(R_idx_te(:,1), R_idx_te(:,2), 1, N, M);
end

% Sanity checks (nnz elements of Rall should be equal to the number of
% indices provided 
if length(R_idx) ~= nnz(Rall) && tetr_split ~= 3
    disp('Problem in Rall.')
elseif length(union(R_idx_te, R_idx_tr, 'rows')) ~= nnz(Rall) && tetr_split == 3
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

%% t-SNE for users' latent factors - Computation

addpath('tSNE_matlab/');
plot_top_20 = 1;      % Show locations of top 20 news_sources
plot_names  = 1;      % Plot names on scatter
plot_subset = 1:1000; % Only plot top 1K sources

% Get index of top 1K sources
[~,I]  = sort(sum(Rall, 2), 1, 'descend');
subidx = I(plot_subset);

% Run t-SNE on subset
ydata = tsne(P(subidx,:));

%% t-SNE for users' latent factors - Plot

% Get ids for known sources to show them in plot
if plot_top_20 == 1
    top_20_str = {'cnn.com', 'bbc.com', 'nytimes.com', 'foxnews.com', ...
                  'washingtonpost.com', 'usatoday.com', ...
                  'theguardian.com', 'dailymail.co.uk', 'chinadaily.com.cn', ...
                  'telegraph.co.uk', 'wsj.com', 'indiatimes.com', 'independent.co.uk', ...
                  'elpais.com', 'lemonde.fr', 'ft.com', 'bostonglobe.com', ...
                  'ap.org', 'afp.com', 'reuters.com', 'yahoo.com', };
              
    top_20_ids = zeros(length(top_20_str),1);

    for ii=1:length(top_20_str)
        id_find = find(strcmp(top_20_str{ii}, names_train));
        if length(id_find) > 0
            top_20_ids(ii) = id_find;
        end
    end
    top_20_ids = top_20_ids(top_20_ids>0);
    plot_idx = ismember(subidx,top_20_ids); % Keep the ones that are part of the plot
else
    plot_idx = subidx;
end

% Scatter plot t-SNE results
% figure;
% scatter(ydata(~plot_idx,1),ydata(~plot_idx,2));
% hold on;
% scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, 'r', 'filled');

figure;
set(gca, 'FontSize', 25);
scatter(ydata(~plot_idx,1),ydata(~plot_idx,2), ...
              'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
hold on
scatter(ydata(plot_idx,1),ydata(plot_idx,2), 300, ...
              'MarkerEdgeColor',[.5 0 0],...
              'MarkerFaceColor',[.9 0 0],...
              'LineWidth',1.5);

plot_names = 2;

% Overlay names
if plot_names == 1
    dx = 0.75; dy = 0.1; % displacement so the text does not overlay the data points
    t = text(ydata(plot_idx,1)+dx, ydata(plot_idx,2)+dy, names_train(subidx(plot_idx)));
    set(t, 'FontSize', 22);
else
    dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
    t = text(ydata(:,1)+dx, ydata(:,2)+dy, names_train(subidx));
    set(t, 'FontSize', 22);
end

xlabel('PC1')
ylabel('PC2')
title('t-SNE projection sources latent space P')

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
dist = @(id, source) log(nnz(source & Rall(id,:)) / sum(source));

recompute_dist = 0;

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
scatter(ydata(:,1), ydata(:,2), 100, dist_ap, 'filled');
hold on;

% Scatter
scatter(ydata(reuters_idx,1), ydata(reuters_idx,2), 300, 'r', 'filled');
scatter(ydata(ap_idx,1),      ydata(ap_idx,2),      300, 'r', 'filled');
% Overlay names
t1 = text(ydata(reuters_idx,1) + dx, ydata(reuters_idx,2) + dy, 'Reuters');
t2 = text(ydata(ap_idx,1)      + dx, ydata(ap_idx,2)      + dy, 'Associated Press');
set(t1, 'FontSize', 22);
set(t2, 'FontSize', 22);
colorbar
xlabel('PC1')
ylabel('PC2')
title('Log-Distance of each source to Associated Press')

figure;
scatter(ydata(:,1), ydata(:,2), 100, dist_reuters, 'filled');
hold on;

% Scatter
scatter(ydata(reuters_idx,1), ydata(reuters_idx,2), 300, 'r', 'filled');
scatter(ydata(ap_idx,1),      ydata(ap_idx,2),      300, 'r', 'filled');
% Overlay names
t1 = text(ydata(reuters_idx,1) + dx, ydata(reuters_idx,2) + dy, 'Reuters');
t2 = text(ydata(ap_idx,1)      + dx, ydata(ap_idx,2)      + dy, 'Associated Press');
set(t1, 'FontSize', 22);
set(t2, 'FontSize', 22);
colorbar
xlabel('PC1')
ylabel('PC2')
title('Log-Distance of each source to Reuters')

%% DBSCAN - Copyright (c) 2015, Yarpiz

addpath('DBSCAN/')

% Configure
epsilon = 2;
MinPts  = 5;
X       = ydata;
% Compute
db      = DBSCAN(X, epsilon, MinPts);
% Plot
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

%% Find recommendation ranking for holdout test event
% top_20 from the dataset

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

%% Ranking - Jay
% Alternative implementation of the recommendation ranking system

% Choose a subset to rank
auto_top_20_ids = subidx(1:end); % Least "popular" sources
unique_te = unique(R_idx_te(:,2));      % Keep only unique events

res = [];
for i=1:length(R_idx_te)
   te_ev = R_idx_te(i,:);               % Get the (source;event) pair
   sp    = P(te_ev(1),:)*Q(:,te_ev(2)); % Compute the score from P;Q matrices for this pair
   
   if any(te_ev(1)==auto_top_20_ids)    % Limit to the selected subset
       cnt = 1;
       for j=1:length(unique_te)
         if i==j; continue; end;
         sn = P(te_ev(1),:)*Q(:,R_idx_te(j,2)); % Get the scores for this sources relative to
                                                % all the events

         if sn>sp;cnt=cnt+1; end;               % Increment ranking if there is a better score
                                                % for another event than
                                                % the one we had selected
       end
       res = [res;cnt];
   end
end

%% Top 20 dustribution

auto_top_20_ids = subidx(1:50); % Least "popular" sources
unique_te = unique(R_idx_te(:,2));      % Keep only unique events

res_20 = [];
for i=1:length(R_idx_te)
   te_ev = R_idx_te(i,:);               % Get the (source;event) pair
   sp    = P(te_ev(1),:)*Q(:,te_ev(2)); % Compute the score from P;Q matrices for this pair
   
   if any(te_ev(1)==auto_top_20_ids)    % Limit to the selected subset
       cnt = 1;
       for j=1:length(unique_te)
         if i==j; continue; end;
         sn = P(te_ev(1),:)*Q(:,R_idx_te(j,2)); % Get the scores for this sources relative to
                                                % all the events

         if sn>sp;cnt=cnt+1; end;               % Increment ranking if there is a better score
                                                % for another event than
                                                % the one we had selected
       end
       res_20 = [res_20;cnt];
   end
end

%% Ranking plot

ranks = res;
figure;
h = hist(res, 500);
scatter(1:1:500, h, 100, ...
              'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .7 .7],...
              'LineWidth',1.5)
set(gca,'xscale','log')
set(gca,'yscale','log')
ylabel('Count')
xlabel('Ranking')
title('Event ranking distribution')
grid on
%set(gca, 'XTickLabel', num2str([1:1:50, 100:100:500, 1000:1000:2000]))

%% Sanity check - Jay
% Check AUC score consistency

unique_te = unique(R_idx_te(:,2)); % Get the unique events in test set

auc = 0;
for i=1:length(R_idx_te)
   te_ev = R_idx_te(i,:);            % Get (source;event) pair
   sp = P(te_ev(1),:)*Q(:,te_ev(2)); % Compute the score for this pair
   
   te_i = te_ev(2); % Get event id
   
   while te_i == te_ev(2)
     rand_i  = randi([1 length(R_idx_te)]);
     te_i = R_idx_te(rand_i, 2);            % Get another random event
   end
   
   sn = P(te_ev(1),:)*Q(:,te_i);            % Compute the score for this new random event
   if sp>sn; auc=auc+1; elseif sp==sn; auc=auc+0.5; end

end

auc = auc / length(R_idx_te);   % Print AUC score
fprintf(['AUC test: ',num2str(auc),'\n']);

%%
alphas = [0.001 0.01 0.05 0.1 0.5 1];
Ks     = [2 5 10 20 30 50];

figure;
colormap('default')
imagesc(heatmap)
ylabel('Learning rate (\alpha)')
xlabel('Latent factors (K)')
set(gca, 'XTickLabel', Ks)
set(gca, 'YTickLabel', alphas)
title('AUC (2e7 iterations, 91421 observations, 5970 holdout)')
colorbar

%% CV


[R_idx_te, M_te, N_te, ids_test , names_test ] = gdelt_weekly_te('data/hashed', reload, subset * tetr_ratio);
[R_idx_tr, M_tr, N_tr, ids_train, names_train] = gdelt_weekly_tr('data/hashed', reload, subset * (1-tetr_ratio));

names = names_train;

M = max(M_te,M_tr);
N = max(N_te,N_tr);

R_idx = union(R_idx_te, R_idx_tr, 'rows');
Rall = sparse(R_idx(:,1), R_idx(:,2), 1);
idx_te = []; % Test indices

% Leave one out per source
for i=1:N_te
    idxs = find(R_idx_te(:,1)==i);      % Find indices corresponding to source
    if ~isempty(idxs)
        rand_idx = randi(length(idxs), 1); % Randomly chose one
        idx_te = [idx_te; idxs(rand_idx)]; % Add it to the test indices
    end
end

% Keep only the heldout test samples
% Create a mask
not_idx_te = zeros(length(R_idx_te),1);
not_idx_te(idx_te) = true;
not_idx_te = ~not_idx_te;

R_idx_tr = [R_idx_tr;R_idx_te(not_idx_te,:)]; % Add the non-heldout events to the training set
R_idx_te(not_idx_te,:) = [];                  % Remove them from the test set, leaving only the
% heldout samples

% Create the Source-Event interaction matrix
Rtr  = sparse(R_idx_tr(:,1), R_idx_tr(:,2), 1, N, M);
Rte  = sparse(R_idx_te(:,1), R_idx_te(:,2), 1, N, M);
%%         
iter = 1e7;
alpha = 0.1;

lambdas = [0.0001;   0.001; 0.01; 0.1; 0.5;  1];
Ks      = [     2;       5;   10;  20;  30; 50];

auc_cv = zeros(length(lambdas), length(Ks));

for cv_iter_lambdas=1:length(lambdas)
    for cv_iter_ks=1:length(Ks)

        % Record auc values
        auc_vals = zeros(iter/100000,1);

        % Initialize low-rank matrices with random values
        P = sigma.*randn(N,Ks(cv_iter_ks)) + mu; % Sources
        Q = sigma.*randn(Ks(cv_iter_ks),M) + mu; % Events

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
            d = (Q(:,ii)-Q(:,ji))*z - lambdas(cv_iter_lambdas)*P(iu,:)';
            P(iu,:) = P(iu,:) + alpha*d';

            % update Q positive
            d = P(iu,:)*z - lambdas(cv_iter_lambdas)*Q(:,ii)';
            Q(:,ii) = Q(:,ii) + alpha*d';

            % update Q negative
            d = -P(iu,:)*z - lambdas(cv_iter_lambdas)*Q(:,ji)';
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
        
        auc_cv(cv_iter_lambdas, cv_iter_ks) = max(auc_vals);
    end
end

%% CV heatmap

figure;
colormap('default')
imagesc(auc_cv)
ylabel('Regularization (\lambda)')
xlabel('Latent factors (K)')
set(gca, 'XTickLabel', Ks)
set(gca, 'YTickLabel', lambdas)
title('AUC (2e7 iterations, 91421 observations, 5968 holdout -- \alpha = 0.1)')
colorbar

%% AUC plot

figure;
xs = [1:1e5:2e7];
ys = auc_vals;
plot(xs,ys, 'LineWidth', 2.5)
hold on
plot(xs, ones(1,length(xs)) .* max(auc_vals), '--', 'LineWidth', 2.5)
grid on
ylabel('AUC')
xlabel('Iteration')
legend('AUC', 'max(AUC)')
title('AUC (2e7 iterations, \alpha=0.1, \lambda=0.01, K=20)')