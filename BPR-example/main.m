% This code is a basic example of one-class Matrix Factorization
% using AUC as a ranking metric and Bayesian Personalized Ranking
% as an optimization procedure (https://arxiv.org/abs/1205.2618).
clear;

iter     =   2e5; % number of iterations
alpha    =  0.05; % learning rate
lambda   =  0.01; % regularizer
sigma    =   0.1; % std for random initialization
mu       =   0.0; % mean for random initialization
K        =    10; % number of latent factors

% Create synthetic dataset
% simulate a one-class collaborative filtering setup
% => positive signals only
N       =  1000; % Users
M       = 10000; % Items
num_pop =     5; % Number of population types
R_idx = gen_data(M,N);

% Split test-train
datalen  = size(R_idx,1);
rp       = randperm(datalen);
pivot    = ceil(datalen/10);
R_idx_te = R_idx(rp(1:pivot),:);
R_idx_tr = R_idx(rp(pivot+1:end),:);

% Create the User-Item interaction matrix
Rall = zeros(N,M);
Rtr  = zeros(N,M);
Rall(R_idx(:,1),R_idx(:,2))      = 1;
Rtr(R_idx_tr(:,1),R_idx_tr(:,2)) = 1;

% Initialize low-rank matrices with random values
P = sigma.*randn(N,K) + mu; % Users
Q = sigma.*randn(K,M) + mu; % Items

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
    
    if mod(step,10000)==0
        
        % Compute the Area Under the Curve (AUC)
        auc = 0;
        for i=1:length(R_idx_te)
            te_i  = randi([1 length(R_idx_te)]);
            te_iu = R_idx_te(i,1);
            te_ii = R_idx_te(i,2);
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

ydata = tsne(P);
scatter(ydata(:,1),ydata(:,2));
