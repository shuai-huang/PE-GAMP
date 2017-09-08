% initialize EXPM parameters
%

function [lambda, beta, omega, optPE] = set_initsPEBM(optPE, Y, A, M, N, T)

%If not defined by user, see if input is complex 
if ~isfield(optPE,'cmplx_in')
    if ~isreal(A.multTr(randn(M,1))) || ~isreal(Y)
        optPE.cmplx_in = true;
    else
        optPE.cmplx_in = false;
    end
end

%If not defined by user, see if output is complex 
if ~isfield(optPE,'cmplx_out')
    if ~isreal(Y)
        optPE.cmplx_out = true;
    else
        optPE.cmplx_out = false;
    end
end
if ~optPE.cmplx_out && ~isreal(Y)
    error('Since measurements are complex, must set optPE.cmplx_out=true')
end

%Initialize all parameters
if ~isfield(optPE,'active_weights')
    L = optPE.L;
else
    L = size(optPE.active_weights,1);
end

%Initialize lambda
if isfield(optPE,'lambda')
    lambda = optPE.lambda;
else
    lambda = 0.1;
end

%Initialize beta
if isfield(optPE,'beta')
    if (size(optPE.beta,2) > 1)
        beta = zeros(1,T,L);
        beta(1,:,:) = optPE.beta';
    else
        beta(1,1,:) = optPE.beta;
    end
else
end

% Initialize Exponential Mixture parameters
omega = zeros(1,1,L);

%Initialize active weights with pre-defined inputs or defaults
if isfield(optPE,'active_weights')
    if (size(optPE.active_weights,2) > 1)
        omega = zeros(1,T,L);
        omega(1,:,:) = optPE.active_weights';
    else
        omega(1,1,:) = optPE.active_weights;
    end
else
end

%Resize all initializations to matrix form for scalar multiplications later
lambda = resize(lambda,N,T);
beta = resize(beta,N,T,L);
omega = resize(omega,N,T,L);

if isfield(optPE, 'noise_var')
	if (size(optPE.noise_var,2) == 1)
		optPE.noise_var = repmat(optPE.noise_var,[M T]);
	else
		optPE.noise_var = repmat(optPE.noise_var,[M 1]);
	end
else
	error('Noise variances missing!')
end


return
