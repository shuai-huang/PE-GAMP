% initialize LP parameters
%

function [lambda, optPE] = set_initsPE(optPE, Y, A, M, N, T)

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


%Initialize lambda
if isfield(optPE,'lambda')
    lambda = optPE.lambda;
else
    lambda = 0.1;
end;

%Resize all initializations to matrix form for scalar multiplications later
lambda = resize(lambda,N,T,1);

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
