%
% MAP parameter estimation of the PE-GAMP for AWGN output channel
%
% The simplifed PE-GAMP where the parameters at different variable nodes are the same
%
% max_ite, max_ite_inner, step_ratio_init should be set appropriately
%
% Shuai Huang, The Johns Hopkins University.
% E-mail: shuang40@jhu.edu
% Date: 10/29/2016
% 

function [noise_var, noise_var_all] = awgn_output_update(Y, Phat, Pvar, noise_var_pre, noise_var_all_pre)

max_ite = 100;               % maximum number of iterations
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

    obj = sqrt(Pvar./(noise_var_pre+Pvar)).*exp(-0.5./(noise_var_pre+Pvar).*(Y-Phat).^2);
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
	noise_var = repmat(noise_var_all, N, T);

	if (norm(noise_var_all-noise_var_all_pre, 'fro')/norm(noise_var_all, 'fro') < tol)
		break;
	end

	noise_var_all_pre = noise_var_all;
	noise_var_pre = noise_var;
end

%fprintf('%5.5f\n', noise_var_all)

end
