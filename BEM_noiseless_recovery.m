% Noiseless recovery using PE-GAMP with the Bernoulli-Exponential mixure input channel and AWGN output channel. The optimal parameters and the sparse signal are estimated using sum-product message passing.

addpath('./BEM')
addpath('./main')

sigma=50;   % over-sampling ratio (to be divided by 100)
rho=75;     % under-sampling ratio (to be divided by 100)

fprintf('%5d   %5d\n', sigma, rho)

N=1000;     % signal length
M=sigma*N/100;  % The number of measurements
S=ceil(rho*M/100);  % The number of nonzero entries, i.e. sparsity level
C=1;

relError_mat=[];
for (i = 1:10)

Phi = randn(M,N);
Phi_norm=sqrt(sum(Phi.^2));
for(j=1:N)
    Phi(:,j)=Phi(:,j)/Phi_norm(j);
end

nonzeroW = exprnd(1, S, C);
ind = randperm(N);
indice = ind(1 : S);
X = zeros(N, C);
X(indice,:) = nonzeroW;
signal = Phi * X;

Y=signal;
A=Phi;

a = omp2(A,Y,1000,1e-6);    % use OMP to initialize
an = a(a>0);

% bernoulli-exponential mixture
cluster_num = 5;

optPE.noise_var = 1e-6;
optPE.lambda=0.1*length(an)/length(a);
optPE.L=cluster_num;
optPE.beta = [0.25/sqrt(var(a(a>0))) 0.5/sqrt(var(a(a>0)))  1/sqrt(var(a(a>0)))  2/sqrt(var(a(a>0)))  4/sqrt(var(a(a>0)))]';
optPE.active_weights = [0.2 0.2 0.2 0.2 0.2]';
optPE.maxPEiter = 100;
optGAMP.nit = 10000;    % this is to ensure best performance, it can be decreased to speed up the recovery
[Xr, PEfin]=BEM_PE_GAMP(Y, A, optPE, optGAMP); 


relError = norm(X-Xr, 'fro')/norm(X, 'fro');

relError_mat=[relError_mat relError];

end

fprintf('Success recovery rate: %d/10\n', sum(relError_mat<1e-3))

