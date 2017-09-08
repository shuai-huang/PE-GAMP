% Noiseless recovery using PE-GAMP with the Laplace input channel and AWGN output channel. The optimal parameter lambda is estimated using sum-product message passing, which is then passed on to the max-product message passing to estimate the sparse signal.

addpath('./LP')
addpath('./main')

sigma=50;   % over-sampling ratio (to be divided by 100)
rho=35;     % under-sampling ratio (to be divided by 100)

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

    % set the appropriate parameters
    optPE.noise_var = 1e-6;
    optPE.lambda= 1/sqrt(var(a)/2);;
    optPE.maxPEiter = 100;
    optGAMP.nit = 1000;
    [Xr, PEfin]=SUM_LP_PE_GAMP(Y, A, optPE, optGAMP); 

    optPE.noise_var = PEfin.noise_var_all;
    optPE.lambda=PEfin.lambda_all;
    optPE.maxPEiter = 1;
    optGAMP.nit = 1000;
    [Xr, PEfin]=MAX_LP_PE_GAMP(Y, A, optPE, optGAMP); 

    relError = norm(X-Xr, 'fro')/norm(X, 'fro');

    relError_mat=[relError_mat relError];

end

fprintf('Success recovery rate: %d/10\n', sum(relError_mat<1e-3))

