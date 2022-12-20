function demo
    warning('off');
    close all;

    % load data 
    disp('Load data...');
    load('USPS_vs_MNIST.mat');
    
    [Xs, Xs_label] = cv_Xs(X_src, Y_src);
    clear X_src; clear Y_src;
    [Xt_train, Xt_train_label, Xt_test, Xt_test_label] = cv_Xt(X_tar, Y_tar);
    clear X_tar; clear Y_tar;
    
    src_n = size(Xs, 1);             % number of source sample for train
    tar_ntra = size(Xt_train, 1);    % number of target sample for train
    tar_ntes = size(Xt_test, 1);     % number of target sample for test
    dim = size(Xs, 2); 
    
    % init parameters
    P = [2, 100, 10, 30, 100, 1, 1e-3, 1e-4]; %[k, C, dim, sigma, lamda, beta, gamma, gammaW]
    param = initParam(P);
                    
    % MTL-GKE
    [acc, result] = MTL_GKE(param, Xs, Xs_label, Xt_train, Xt_test, Xt_train_label, Xt_test_label);
    acc             
end


%%

function param = initParam(P)
    P=[1, 100, 50, 2, 1000, 0.001, 1e-6, 1e-5];  %[k, C, dim, sigma, lamda, beta, gamma, gammaW]
    param.k=P(1);
    param.num_constraints=P(2);
    param.dim=P(3);
    param.sigma = P(4);             
    param.lamda=P(5);               
    param.beta = P(6);
    param.gamma=P(7);               
    param.gammaW=P(8);
    param.a=5;
    param.b=95;
    param.eplsion=1e-7;             
end

function [acc, result] = MTL_GKE(param, Xs, Xs_label, Xt_train, Xt_test, Xt_train_label, Xt_test_label)
    %% paramter
    tic
    a = param.a; 
    b = param.b;
    num_constraints = param.num_constraints;
    k = param.k;
    dim = param.dim;
    
    %% load data
    X = [Xs;Xt_train];                  % all data for train 
    y = [Xs_label';Xt_train_label'];
    Xtest = Xt_test;                    % all data for test 
    ytest = Xt_test_label';
    
    %% create model for pseudo labeling
    model = fitcecoc(Xt_train, Xt_train_label, 'weight', []);
    Cls = predict(model, Xtest);
    
    %% pca data dim reduce
    Xr = [X;Xtest]; % all the data for train and test (for PCA)
    reduced_X = PCA_reduce(Xr, dim);
    
    %% normalization
    X = reduced_X(1:size(X,1),:); 
    Xtest = reduced_X(size(X,1)+1:size(Xr), :);
    
    %% Compute weights
    ntra_s = size(Xs);
    ntra_t = size(Xt_train);
    x_source = X(1:ntra_s,:);
    x_target = [X(ntra_s+1:ntra_s+ntra_t,:);Xtest];
    
    %% Compute distance extremes
    [l, u] = ComputeDistanceExtremes(X, a, b);
    
    % Generate constraint point pairs
    C = GetConstraints(y, num_constraints, l, u);
    Xci = X(C(:,1),:);
    yci = y(C(:,1), :);
    Xcj = X(C(:,2),:); 
    ycj = y(C(:,2), :);
    
    %% Optimization
    d = size(X, 2); 
    p = num_constraints; 
    sd_tra = X(1:ntra_s, :); 
    td_tra = X(ntra_s+1:ntra_s+ntra_t, :); 
    
    
    options.k = 100;                % subspace bases
    options.delta =1;               % regularizer
    options.T = 50;                 % iterations,
    options.ker = 'linear';         % kernel type
    options.TITlambda = 0.1;
    options.TITgamma = 1;
    Goptions = [];
    Goptions.model = 'sig';
    Goptions.k = 8;
    Goptions.NeighborMode = 'KNN';
    Goptions.bNormalized = 1;
    Goptions.WeightMode = 'Cosine';   %'HeatKernel';%'Cosine';
    Goptions.t = 0.5;
    Goptions.NeighborMode = 'Supervised';
    Goptions.gnd = [Xs_label';Xt_train_label';Cls];
    Goptions.weight = [];
    
    
    [A, result] = optimization(C, Xci, Xcj, param, sd_tra, td_tra, options, Goptions, Cls, reduced_X, x_target, y, Xr);
    
    
    %% Predict
    preds = KNN(y, X, A, k, Xtest);
    acc = sum(preds==ytest)/size(ytest, 1);
    acc = acc(1, 1);
    toc
end

%% optimization
function [At, result] = optimization(C, Xci, Xcj, param, sd_tra, td_tra, options, Goptions, Cls, reduced_X, x_target, y, Xr)
    eplsion = param.eplsion;         
    sigma = param.sigma;            % penalti coeficient rho          
    lamda = param.lamda; 
    beta = param.beta;     
    gamma = param.gamma; 
    gammaW = param.gammaW/sigma;   
    TITlambda = options.TITlambda;
    delta = options.delta;
    TITgamma = options.TITgamma;
    TITk = options.k;
    ker = options.ker;
    A0 = eye(size(Xci,2),size(Xci,2));
    E = A0;
    At = A0;
    
    ns = size(sd_tra, 1);  
    nst = size(td_tra, 1); 
    n = size(reduced_X, 1);
    nt = size(x_target, 1);
    e = zeros(ns+nst, 1);     
    e(1:ns,:) = 1;  
    iter = 0;    
    convA = 1000;
    weight0 = zeros(ns+nst, 1);
    weight = weight0;
    
    K = kernel(ker, Xr', [], TITgamma);              % compute kernel k (linear transformation of Xr)
    H = eye(n)-1/(n)*ones(n,n);                   % compute centering matrix H
    e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];       %
    M = e*e';                                     %
    M = M/norm(M, 'fro');                          % compute MMD matrix
    G = speye(n);                                 % initialize G
    R = 0.01*eye(n);                                % compute betha*I
    R(1:ns, 1:ns) = 0;
    
    %%compute Laplacian matrix
    if strcmp(Goptions.model, 'sig')
        W= constructW(reduced_X, Goptions);
        D = diag(full(sum(W,2)));                 % Degree of each vertex
        W1 = -W;                                  % Laplacian matrix
        for i=1:size(W1,1)
            W1(i,i) = W1(i,i) + D(i,i); 
        end
    else
        [Ww,Wb] = ConG(Goptions.gnd_train, Goptions, X', 'srge');
        W1 = Ww-options.gamma*Wb;
    end
    
    
    while convA > eplsion && iter < 1
        
        %%update p
        V0 = ones(size(K*(M+TITlambda*W1-R)*K'+delta*G,1), 1);
        [P,~] = eigs(K*(M+TITlambda*W1-R)*K'+delta*G, K*H*K', TITk, 'SM', struct('V0', V0));
        G = diag(sparse(1./(sqrt(sum(P.^2,2)+eps))));
        Z = P'*K;
        
        Goptions.gnd = [y;Cls];
        
        if strcmp(Goptions.model,'sig')
            W= constructW(reduced_X, Goptions);
            D = diag(full(sum(W,2)));
            W1 = -W;
            for i=1:size(W1,1)
                W1(i,i) = W1(i,i) + D(i,i);
            end
        else
            [Ww,Wb] = ConG(Goptions.gnd_train, Goptions,X', 'srge');
            W1 = Ww-options.gamma*Wb;
        end
        
        for c=1:length(unique(Cls))
            idt = Cls==c;
            Zs = Z(:,1:ns)';
            Zt = Z(:,ns+nst+1:n)';
            Tsamps = Zt(idt, :);
            [idx, dist] = knnsearch(Zs, Tsamps, 'distance', 'cosine', 'k', 10);
            idx = idx(:);
            for i=1:length(idx)
               weight(idx(i,:), :) = weight(idx(i,:),:)+1;
            end
        end
        weight = weight./sum(weight);
        
        %% updata A
        sumA = zeros(size(Xci,2), size(Xci,2));
        pair_weights = weight(C(:,1),:).*weight(C(:,2),:).*C(:,3);
        for i=1:size(Xci,1)
            vij = Xci(i,:)-Xcj(i,:);
            sumA = sumA+A0*vij'*vij*pair_weights(i)*C(i,3);
        end
        At = At-gamma*(2*beta*sumA+2*A0);           %Equation 18
        
        %% compute threshold
        convA = norm(At-A0);
        convW = norm(weight-weight0);
        A0 = At;
        iter = iter+1;
        sumA = 0;
        for k=1:size(Xci,1)
            i = C(k,1); 
            j=C(k,2); 
            deta_ij = C(k,3);
            vij = Xci(k,:)-Xcj(k,:);  
            vijA = vij*At'; dij=vijA*vijA';
            sumA = sumA + weight(i)*weight(j)*dij*deta_ij;
        end
        f1(iter) = trace(At'*At);
        f2(iter) = trace(P'*K*(M+TITlambda*W1-R)*K'*P)+norm(P);
        f3(iter) = beta*sumA;
        f = f1(iter)+f2(iter)+f3(iter);               % Equation 13
        convWs(iter) = convW;   
        convAs(iter) = convA;
        As(iter).A = At;
        ws(iter,:) = weight;
        fs(iter) = f;
        if iter >= 2
            fd(iter-1) = fs(iter)-fs(iter-1);
            Ad(iter-1) = norm(As(iter).A -As(iter-1).A);
            wd(iter-1) = norm(ws(iter,:)-ws(iter-1,:));
        else
            fd(1) = 1;
            Ad(1) = 1;
            wd(1) = 1;
        end
    end
    time = toc;
    result.At = At;
    result.wt = weight;
    result.w0 = weight0;
    result.Ad = Ad;
    result.wd = wd;
    result.convAs = convAs;
    result.convWs = convWs;
    result.fs = fs;
    result.fd = fd;
    result.f1 = f1;
    result.f2 = f2;
    result.f3 = f3;
    result.time = time;
end
