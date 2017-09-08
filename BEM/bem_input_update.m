%
% MAP parameter estimation of the PE-GAMP for Bernoulli-Exponential mixture input channel
%
% The simplified PE-GAMP where the parameters at different variable nodes are the same
%
% max_ite, max_ite_inner, step_ratio_init should be set appropriately
%
% lambda - Bernoulli distribution parameter [0,1]
% beta   - Exponential distribution parameter (0, infty)
% omega  - Exponential mixture weights
%
% Shuai Huang, The Johns Hopkins University.
% E-mail: shuang40@jhu.edu
% Date: 09/03/2017
% 

function [lambda, lambda_all, beta, beta_all, omega, omega_all] = bem_input_update(Rhat, Rvar, lambda_pre, lambda_all_pre, beta_pre, beta_all_pre, omega_pre, omega_all_pre)

max_ite = 100;               % maximum number of iterations
                            % remember to tune this for best performance 10~100
tol = 1e-6;                 % convergence criteiron
max_ite_inner = 1000;       % maximum number of iterations in the inner loop
step_ratio_init = 0.01;     % initial step size, set to a relatively small number for stability
dec_rate = 0.9;             % the decreasing rate of the step ratio
step_ratio_min = 1e-6;      % the minimum step size

% The Bernoulli weights limit
lambda_min = 1e-12;         % mimimum lambda value
lambda_max = 1;             % maximum lambda value

% The minimum and maximum Exponential distribution parameter beta
beta_all_min = 1e-12;
beta_all_max = 1e10;    % just set to some relatively large number

% The number of mixiture components
L = size(omega_all_pre, 2);

%Calcualte Problem dimensions
[N, T] = size(Rhat);

alpha_pre = log(omega_pre);
alpha_all_pre = log(omega_all_pre);

% save the old parameters
lambda_pre_old = lambda_pre;
lambda_all_pre_old = lambda_all_pre;
beta_pre_old = beta_pre;
beta_all_pre_old = beta_all_pre;
omega_pre_old = omega_pre;
omega_all_pre_old = omega_all_pre;
alpha_pre_old = alpha_pre;
alpha_all_pre_old = alpha_all_pre;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simplify stuff here, start from the same lambda_all %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig = sqrt(Rvar);                           % Gaussian prod std dev

% let gamma_one=log(1-lambda), gamma_two=log(lambda)
% lambda should be updated at last
% simplify stuff here, start from the same lambda_all
c = exp(-Rhat.^2./(2*Rvar));
d = zeros(N,T,L);
for (idx_mix = 1:L)
    beta_pre_mix = beta_pre(:,:,idx_mix);
    omega_pre_mix = omega_pre(:,:,idx_mix);
    omega_beta_pre_mix = omega_pre_mix.*beta_pre_mix;

    muU = Rhat - beta_pre_mix.*Rvar;              % Upper integral mean
    muU_over_sig = muU ./ sig;

    C_U = erfcx(-muU_over_sig / sqrt(2));

    SC_U = sqrt(pi/2)*C_U;
    d(:,:,idx_mix) = beta_pre_mix.*sig.*(SC_U);
end

d_omega = zeros(N,T);
for (i=1:L)
    d_omega = d_omega + d(:,:,i)*omega_all_pre(i);
end

ite=1;
while (ite<=max_ite)
    ite=ite+1;
    lambda_der_first = (d_omega-c)./((1-lambda_all_pre)*c+lambda_all_pre*d_omega);
    lambda_der_first(isnan(lambda_der_first)) = 1/lambda_all_pre;

    obj_old = ((1-lambda_all_pre)*c+lambda_all_pre*d_omega)./(c+d_omega);
    obj_old(isnan(obj_old)) = lambda_all_pre;
    [umd, umdi]=max(d_omega);

    step_ratio = 0;
    if (sum(lambda_der_first)>0)
        step_ratio = step_ratio_init;
    else
        step_ratio = -step_ratio_init;
    end
    if (lambda_all_pre==1)
        step_ratio = -step_ratio_init;
    end
    ite_inner = 0;
    while (ite_inner<=max_ite_inner)
        ite_inner = ite_inner+1;
        lambda_all = lambda_all_pre*(1+step_ratio);
        lambda_all = max(lambda_min, lambda_all);
        lambda_all = min(lambda_max, lambda_all);
        obj_new = ((1-lambda_all)*c + lambda_all*d_omega)./(c+d_omega);
        obj_new(isnan(obj_new)) = lambda_all;
        
        if (abs(step_ratio)<step_ratio_min)
            lambda_all = lambda_all_pre;
            break;
        end
        if (sum(log(obj_new))>sum(log(obj_old))) 
            break;
        else
            step_ratio = step_ratio*dec_rate;
        end
    end

    lambda_all = max(lambda_min, lambda_all);
    lambda_all = min(lambda_max, lambda_all);
    lambda = repmat(lambda_all, N,T);

    if (norm(lambda_all-lambda_all_pre, 'fro')/norm(lambda_all, 'fro') < tol)
        break;
    end

    lambda_all_pre = lambda_all;
    lambda_pre = lambda;
end

% Reinstate the lambda parameter values
lambda_all_pre = lambda_all_pre_old;
lambda_pre = lambda_pre_old;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update beta, the exponential distribution parameter %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_all_out = beta_all_pre;
beta_out = beta_pre;

for (ii=1:L)
ite = 1;
% Compute the following part once
der_first_denominator_mix = zeros(N,T,L);
muU_over_sig_all = zeros(N,T,L);
for (idx_mix=1:L)

    beta_pre_mix = beta_pre(:,:,idx_mix);
    omega_pre_mix = omega_pre(:,:,idx_mix);
    omega_beta_pre_mix = omega_pre_mix.*beta_pre_mix;

    muU = Rhat - beta_pre_mix.*Rvar;              % Upper integral mean
    muU_over_sig = muU ./ sig;
    muU_over_sig_all(:,:,idx_mix) = muU_over_sig;

    C_U = erfcx(-muU_over_sig / sqrt(2));

    SC_U = sqrt(pi/2)*C_U;
    der_first_denominator_mix(:,:,idx_mix) = omega_beta_pre_mix.*sig.*(SC_U);
end

while(ite<=max_ite)

    ite = ite+1;

    der_first_numerator = zeros(N,T);
    idx_mix=ii;
    beta_pre_mix = beta_pre(:,:,idx_mix);
    omega_pre_mix = omega_pre(:,:,idx_mix);
    omega_beta_pre_mix = omega_pre_mix.*beta_pre_mix;

    muU = Rhat - beta_pre_mix.*Rvar;              % Upper integral mean
    muU_over_sig = muU ./ sig;
    muU_over_sig_all(:,:,idx_mix) = muU_over_sig;

    C_U = erfcx(-muU_over_sig / sqrt(2));

    SC_U = sqrt(pi/2)*C_U;
    der_first_numerator = omega_pre_mix.*sig.*(SC_U) + omega_beta_pre_mix.*(-Rvar-muU.*sig.*SC_U);
    der_first_denominator_mix(:,:,idx_mix) = omega_beta_pre_mix.*sig.*(SC_U);

    der_first_numerator = der_first_numerator*lambda_all_pre;
    der_first_denominator = (1-lambda_all_pre)*c + lambda_all_pre*sum(der_first_denominator_mix, 3);
    der_first = der_first_numerator ./ der_first_denominator;
    
    nan_marker = isnan(der_first);
    der_first(nan_marker) = (omega_pre_mix(nan_marker).*sig(nan_marker)+omega_beta_pre_mix(nan_marker).*(-muU(nan_marker).*sig(nan_marker)) )*lambda_all_pre  ./ (lambda_all_pre*omega_beta_pre_mix(nan_marker).*sig(nan_marker)) ;

    obj_old = der_first_denominator;

    step_ratio = 0;
    if (sum(der_first)>0)&&(beta_all_pre(ii)>0)
        step_ratio = step_ratio_init;
    elseif (sum(der_first)>0)&&(beta_all_pre(ii)<0)
        step_ratio = -step_ratio_init;
    elseif (sum(der_first)<0)&&(beta_all_pre(ii)>0)
        step_ratio = -step_ratio_init;
    else
        step_ratio = step_ratio_init;
    end

    ite_inner = 0; 
    beta_all = beta_all_pre;
    while(ite_inner<=max_ite_inner)
        ite_inner=ite_inner+1;
        beta_all(ii) = beta_all_pre(ii)*(1+step_ratio);
        der_first_denominator_mix_tmp = der_first_denominator_mix;
        muU_over_sig_all_tmp = muU_over_sig_all;
        idx_mix=ii;
            beta_pre_mix = beta_all(idx_mix);
            omega_pre_mix = omega_pre(:,:,idx_mix);
            omega_beta_pre_mix = omega_pre_mix*beta_pre_mix;

            muU = Rhat - beta_pre_mix*Rvar;              % Upper integral mean
            muU_over_sig = muU ./ sig;
            muU_over_sig_all_tmp(:,:,idx_mix) = muU_over_sig;

            C_U = erfcx(-muU_over_sig / sqrt(2));

            SC_U = sqrt(pi/2)*C_U;
            der_first_denominator_mix_tmp(:,:,idx_mix) = omega_beta_pre_mix.*(SC_U);
        
        der_first_denominator_tmp = (1-lambda_all_pre)*c +lambda_all_pre*sum(der_first_denominator_mix_tmp,3);
        obj_new = der_first_denominator_tmp;

        if (abs(step_ratio)<step_ratio_min)
            beta_all(ii) = beta_all_pre(ii);
            break;
        end
        obj_ratio = obj_old./obj_new;
        if (any(isnan(obj_ratio)))
            idx_nan = 1:N;
            idx_nan = idx_nan(isnan(obj_ratio));
            for (idxn = idx_nan)
                muU_os_nan = squeeze(muU_over_sig_all(idxn,:,:))';
                muU_os_tmp_nan = squeeze(muU_over_sig_all_tmp(idxn,:,:))';
                muU_os_nan_seq = [muU_os_nan muU_os_tmp_nan];
                [Um, Ui] = max(muU_os_nan_seq);
                muU_os_nan_sq_new = (muU_os_nan.^2 - Um^2)./2;
                muU_os_tmp_nan_sq_new = (muU_os_tmp_nan.^2 - Um^2)./2;
                obj_num_nan = omega_all_pre.*beta_all_pre.*exp(muU_os_nan_sq_new).*normcdf(muU_os_nan);
                obj_den_nan = omega_all_pre.*beta_all.*exp(muU_os_tmp_nan_sq_new).*normcdf(muU_os_tmp_nan);
                obj_nan = sum(obj_num_nan)/sum(obj_den_nan);
                obj_ratio(idxn) = obj_nan;
            end
        end

        if (sum(log(obj_ratio))<0) 
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end 
    end

    beta_all = max(beta_all_min, beta_all);
    beta_all = min(beta_all_max, beta_all);

    beta = zeros(N,T,L);
    for (idx_mix=1:L)
        beta(:,:,idx_mix) = repmat(beta_all(idx_mix), N, T);
    end

    if (norm(beta_all(ii)-beta_all_pre(ii), 'fro')/norm(beta_all(ii), 'fro')<tol)
        break;
    end

    beta_all_pre = beta_all;
    beta_pre = beta;
end
beta_all_out(ii) = beta_all(ii);
beta_out(:,:,ii) = beta(:,:,ii);
beta_all_pre = beta_all_pre_old;
beta_pre = beta_pre_old;
end
% Output the maximizing beta
beta_all = beta_all_out;
beta = beta_out;

% Reinstate the mu values
beta_all_pre = beta_all_pre_old;
beta_pre = beta_pre_old;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update the omega, the Exponential mixture weights %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_out = alpha_pre;
alpha_all_out = alpha_all_pre;

for (ii=1:L)

muU_over_sig_all = zeros(N,T,L);
SC_U_all = zeros(N,T,L);
for (idx_mix=1:L)
    beta_pre_mix = beta_pre(:,:,idx_mix);
    muU = Rhat - beta_pre_mix.*Rvar;              % Upper integral mean
    muU_over_sig = muU ./ sig;
    muU_over_sig_all(:,:,idx_mix) = muU_over_sig;
    C_U = erfcx(-muU_over_sig / sqrt(2));
    SC_U_all(:,:,idx_mix) = sqrt(pi/2)*C_U;
end

ite = 1;
% once one omega changes, all omega would change
while(ite<=max_ite)
    
    ite = ite+1;
    der_first_numerator = zeros(N,T,L);
    der_first_denominator_mix = zeros(N,T,L);
    for (idx_mix=1:L)
        beta_pre_mix = beta_pre(:,:,idx_mix);
        omega_pre_mix = omega_pre(:,:,idx_mix);
        omega_beta_pre_mix = omega_pre_mix.*beta_pre_mix;

        SC_U = SC_U_all(:,:,idx_mix);
        
        der_first_numerator(:,:,idx_mix) = beta_pre_mix.*(SC_U);
        der_first_denominator_mix(:,:,idx_mix) = omega_beta_pre_mix.*(SC_U);
    end
    der_first_numerator = lambda_all_pre*der_first_numerator;
    der_first_denominator = (1-lambda_all_pre)*c + lambda_all_pre*sum(der_first_denominator_mix, 3);
    alpha_der_first = zeros(N,T,L);
    for (idx_mix=1:L)
    	alpha_der_first_ii = der_first_numerator(:,:,idx_mix) ./ der_first_denominator;
    	if (any(isnan(alpha_der_first_ii)))
		    idx_nan=1:N;
		    idx_nan=idx_nan(isnan(alpha_der_first_ii));
		    beta_pre_mix = beta_pre(:,:,idx_mix);
		    omega_pre_mix = omega_pre(:,:,idx_mix);
		    omega_beta_pre_mix = omega_pre_mix.*beta_pre_mix;
		    alpha_der_first_ii(idx_nan) = beta_pre_mix(idx_nan)./omega_beta_pre_mix(idx_nan);
		end
		if (idx_mix==ii)
			alpha_der_first(:,:,idx_mix) = alpha_der_first_ii * (omega_all_pre(idx_mix)-omega_all_pre(idx_mix)^2);
		else
			alpha_der_first(:,:,idx_mix) = alpha_der_first_ii * (-omega_all_pre(ii)*omega_all_pre(idx_mix));
		end
    end
    alpha_der_first = sum(alpha_der_first, 3);
    
    
    obj_old = der_first_denominator;
    step_ratio=0;
    if (sum(alpha_der_first)>0)&&(alpha_all_pre(ii)>0)
        step_ratio = step_ratio_init;
    elseif (sum(alpha_der_first)>0)&&(alpha_all_pre(ii)<0)
        step_ratio = -step_ratio_init;
    elseif (sum(alpha_der_first)<0)&&(alpha_all_pre(ii)>0)
        step_ratio = -step_ratio_init;
    else
        step_ratio = step_ratio_init;
    end

    ite_inner = 0;
    alpha_all = alpha_all_pre;
    while(ite_inner<max_ite_inner)
        ite_inner = ite_inner +1;
        alpha_all(ii) = alpha_all_pre(ii)*(1+step_ratio);
        alpha_all_exp = exp(alpha_all);
        omega_all = alpha_all_exp/sum(alpha_all_exp);

        der_first_denominator_mix_tmp = zeros(N,T,L);
        for (idx_mix=1:L)
            beta_pre_mix = beta_pre(:,:,idx_mix);
            omega_pre_mix = omega_all(idx_mix);
            omega_beta_pre_mix = omega_pre_mix*beta_pre_mix;

            SC_U = SC_U_all(:,:,idx_mix);;
            der_first_denominator_mix_tmp(:,:,idx_mix) = omega_beta_pre_mix.*(SC_U);
        end
        der_first_denominator_tmp = (1-lambda_all_pre)*c + lambda_all_pre*sum(der_first_denominator_mix_tmp, 3);

        obj_new = der_first_denominator_tmp;
        if (abs(step_ratio)<step_ratio_min)
            alpha_all(ii) = alpha_all_pre(ii);
            break;
        end
        obj_ratio = obj_old./obj_new;
        if (any(isnan(obj_ratio)))
            idx_nan = 1:N;
            idx_nan = idx_nan(isnan(obj_ratio));
            for (idxn = idx_nan)
                muU_os_nan = squeeze(muU_over_sig_all(idxn,:,:))';
                [Um, Ui] = max(muU_os_nan);
                muU_os_nan_sq_new = (muU_os_nan.^2 - Um^2)./2;
                obj_num_nan = omega_all_pre.*beta_all_pre.*exp(muU_os_nan_sq_new).*normcdf(muU_os_nan);
                obj_den_nan = omega_all.*beta_all_pre.*exp(muU_os_nan_sq_new).*normcdf(muU_os_nan);
                obj_nan = sum(obj_num_nan)/sum(obj_den_nan);
                obj_ratio(idxn) = obj_nan;
            end
        end

        
        if (sum(log(obj_ratio))<0) 
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end
    end

    alpha_all_exp = exp(alpha_all);
    omega_all = alpha_all_exp/sum(alpha_all_exp);
    omega = zeros(N,T,L);
    alpha = zeros(N,T,L);
    for (i=1:L)
        omega(:,:,i)=repmat(omega_all(i),N,T);
        alpha(:,:,i)=repmat(alpha_all(i),N,T);
    end

    if (norm(omega_all(ii)-omega_all_pre(ii), 'fro')/norm(omega_all(ii), 'fro')<tol)
        break;
    end

    omega_all_pre = omega_all;
    omega_pre = omega;

    alpha_all_pre = alpha_all;
    alpha_pre = alpha;

end

alpha_all_out(ii) = alpha_all(ii);
alpha_out(:,:,ii) = alpha(:,:,ii);

alpha_all_pre = alpha_all_pre_old;
alpha_pre = alpha_pre_old;

omega_all_pre = omega_all_pre_old;
omega_pre = omega_pre_old;

end

% Output the maximizing alpha and omega
alpha_all = alpha_all_out;
alpha = alpha_out;

alpha_all_exp = exp(alpha_all);
omega_all = alpha_all_exp/sum(alpha_all_exp);
omega = zeros(N,T,L);
alpha = zeros(N,T,L);
for (i=1:L) 
    omega(:,:,i)=repmat(omega_all(i),N,T);
    alpha(:,:,i)=repmat(alpha_all(i),N,T);
end


%fprintf('%5.5f\n', lambda_all)
%for (i=1:L)
%fprintf('%5.5f  %5.5f\n',  omega_all(i), beta_all(i))
%end


end
