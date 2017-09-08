% Noiseless recovery using PE-GAMP with the Bernoulli-Gaussian mixture input channel and AWGN output channel. The optimal parameters and the sparse signal are estimated using sum-product message passing.

addpath('./BGM') 
addpath('./main')

sigma=50;   % over-sampling ratio (to be divided by 100)
rho=55;     % under-sampling ratio (to be divided by 100)

fprintf('%5d   %5d\n', sigma, rho)

N=1000;     % signal length
M=sigma*N/100;  % The number of measurements
S=ceil(rho*M/100);  % The number of nonzero entries, i.e. sparsity level
C=1;

relError_mat=[];
for (i = 1:10)

    fprintf('Recovering %d-th sample\n', i)

    Phi = randn(M,N);
    Phi_norm=sqrt(sum(Phi.^2));
    for(j=1:N)
        Phi(:,j)=Phi(:,j)/Phi_norm(j);
    end

    nonzeroW = randn(S, C);
    ind = randperm(N);
    indice = ind(1 : S);
    X = zeros(N, C);
    X(indice,:) = nonzeroW;
    signal = Phi * X;

    Y=signal;
    A=Phi;

    a = omp2(A,Y,1000,1e-6);    % use OMP to initialize
    an = a(a>0);

    % bernoulli-gaussian mixture
    cluster_num = 5;

    idx_kmeans = kmeans(an,cluster_num,'MaxIter', 1000);
    active_mean=[];
    active_var=[];
    active_weights=[];
    for (ii=1:cluster_num)
        active_mean = [active_mean mean(an(idx_kmeans==ii))];
        active_var = [active_var var(an(idx_kmeans==ii))];
        active_weights = [active_weights length(an(idx_kmeans==ii))/length(an)];
    end

    % choose the appropriate parameters
    optPE.active_mean=active_mean';
    optPE.active_var=active_var'+1e-6;
    optPE.active_weights=active_weights';

    optPE.noise_var = 1e-6;
    optPE.lambda=0.1*length(an)/length(a);
    optPE.maxPEiter = 100;
    optPE.L=cluster_num;
    optGAMP.nit = 1000;
    [Xr, PEfin]=BGM_PE_GAMP(Y, A, optPE, optGAMP); 


    relError = norm(X-Xr, 'fro')/norm(X, 'fro');

    relError_mat=[relError_mat relError];

end

fprintf('Success recovery rate: %d/10 \n', sum(relError_mat<1e-3))

