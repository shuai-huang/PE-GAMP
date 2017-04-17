%
% MAP parameter estimation of the PE-GAMP for AWGN output channel
%
% The PE-GAMP where the parameters at different variable nodes are different
%
% Shuai Huang, The Johns Hopkins University.
% E-mail: shuang40@jhu.edu
% Date: 10/29/2016
% 

function [noise_var, noise_var_all] = awgn_output_update(Y, Phat, Pvar, noise_var_pre, noise_var_all_pre)

max_ite = 10;               % maximum number of iterations
tol = 1e-6;                 % convergence criteiron
max_ite_inner = 1000;       % maximum number of iterations in the inner loop
step_ratio_init = 0.01;     % initial step size
dec_rate = 0.9;             % the decreasing rate of the step ratio
step_ratio_min = 1e-6;      % the minimum step size
noise_var_min = 1e-12;      % minimum noise variance


%Calcualte Problem dimensions
[N, T] = size(Phat);

% simplify stuff here, start from the same lambda_all
Phat_output = Phat;
Pvar_output = Pvar;

ite = 1;
while(ite<=max_ite)

    ite = ite+1;
    der_first = (Y-Phat).^2./(2*(noise_var_all_pre+Pvar).^2) - 0.5./(noise_var_all_pre+Pvar);

    obj = sqrt(Pvar./(noise_var_all_pre+Pvar)).*exp(-0.5./(noise_var_all_pre+Pvar).*(Y-Phat).^2);
    step_ratio = 0;
    if (sum(der_first)>0)
        step_ratio = step_ratio_init;
    else
        step_ratio = -step_ratio_init;
    end
    ite_inner = 0;
    while(ite_inner<=max_ite_inner)
        ite_inner=ite_inner+1;
        noise_var_all = noise_var_all_pre*(1+step_ratio);
        obj_new  = sqrt(Pvar./(noise_var_all+Pvar)).*exp(-0.5./(noise_var_all+Pvar).*(Y-Phat).^2);
        if (abs(step_ratio)<step_ratio_min)
            noise_var_all = noise_var_all_pre;
            break;
        end
        
        if (sum(log(obj_new(obj_new~=0)))>sum(log(obj(obj~=0)))) 
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end
    end

	noise_var_all = max(noise_var_min, noise_var_all);
	%noise_var = repmat(noise_var_all, N, T);

	if (norm(noise_var_all-noise_var_all_pre, 'fro')/norm(noise_var_all, 'fro') < tol)
		break;
	end

	noise_var_all_pre = noise_var_all;
	%noise_var_pre = noise_var;
end

% Output the noise_var value
% Compute first order derivative at the maximizing noise_var_all
der_first = (Y-Phat).^2./(2*(noise_var_all+Pvar).^2) - 0.5./(noise_var_all+Pvar);
% Compute second order derivative at the maximizing lambda_all
der_second = -(Y-Phat).^2./((noise_var_all+Pvar).^3) + 0.5./((noise_var_all+Pvar).^2);
der_second_sum = sum(der_second);
% Compute the lambda value at different nodes
noise_var = repmat(noise_var_all, N, T);
noise_var = noise_var + der_first./(der_second_sum-der_second);

if (any(isinf(noise_var)))
    idx_inf = find(isinf(noise_var));
    noise_var(idx_inf) = noise_var_all;
end

if (any(isnan(noise_var)))
    idx_nan = find(isnan(noise_var));
    noise_var(idx_nan) = noise_var_all;
end

noise_var = max(noise_var_min, noise_var);


fprintf('%5.5f\n', noise_var_all)

end
