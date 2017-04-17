%
% MAP parameter estimation of the PE-GAMP for Bernoulli-Gaussian mixture input channel
%
% The simplified PE-GAMP where the parameters at different variable nodes are the same
%
% lambda - Bernoulli distribution parameter [0,1]
% omega  - Gaussian mixture weights
% mu     - Gaussian mixture means
% phi    - Gaussian mixture variances
%
% Shuai Huang, The Johns Hopkins University.
% E-mail: shuang40@jhu.edu
% Date: 10/29/2016
% 

function [lambda, lambda_all, omega, omega_all, mu, mu_all, phi, phi_all] = bgm_input_update(Rhat, Rvar, lambda_pre, lambda_all_pre, omega_pre, omega_all_pre, mu_pre, mu_all_pre, phi_pre, phi_all_pre)

max_ite = 10;               % maximum number of iterations
tol = 1e-6;                 % convergence criteiron
max_ite_inner = 1000;       % maximum number of iterations in the inner loop
step_ratio_init = 0.01;     % initial step size, set to a relatively small number for stability
dec_rate = 0.9;             % the decreasing rate of the step ratio
step_ratio_min = 1e-6;      % the minimum step size

% The Bernoulli weights limit
lambda_min = 1e-12;         % mimimum lambda value
lambda_max = 1;             % maximum lambda value

% The minimum Gaussian mixture variance
phi_all_min = 1e-6;

% The number of mixiture components
L=size(omega_all_pre, 2);

%Calcualte Problem dimensions
[N, T] = size(Rhat);


alpha_pre = log(omega_pre);
alpha_all_pre = log(omega_all_pre);

% save the old parameters
lambda_pre_old = lambda_pre;
lambda_all_pre_old = lambda_all_pre;
mu_pre_old = mu_pre;
mu_all_pre_old = mu_all_pre;
phi_pre_old = phi_pre;
phi_all_pre_old = phi_all_pre;
omega_pre_old = omega_pre;
omega_all_pre_old = omega_all_pre;
alpha_pre_old = alpha_pre;
alpha_all_pre_old = alpha_all_pre;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simplify stuff here, start from the same lambda_all %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c = exp(-Rhat.^2./(2*Rvar));
d = zeros(N,T,L);
for (i=1:L)
    d(:,:,i) = sqrt(abs(Rvar./(phi_all_pre(i)+Rvar))).*exp(-0.5./(phi_all_pre(i)+Rvar).*(mu_all_pre(i)-Rhat).^2);
end

d_omega = zeros(N,T);
for (i=1:L)
    d_omega = d_omega + d(:,:,i)*omega_all_pre(i);
end

ite=1;
while (ite<=max_ite)
    ite=ite+1;
    lambda_der_first = (d_omega-c)./((1-lambda_all_pre)*c+lambda_all_pre*d_omega);

    obj = (1-lambda_all_pre)*c+lambda_all_pre*d_omega;
    step_ratio = 0;
    if (sum(lambda_der_first)>0)
        step_ratio = step_ratio_init;
    else
        step_ratio = -step_ratio_init;
    end
    ite_inner = 0;
    while (ite_inner<=max_ite_inner)
        ite_inner = ite_inner+1;
        lambda_all = lambda_all_pre*(1+step_ratio);
        lambda_all = max(lambda_min, lambda_all);
        lambda_all = min(lambda_max, lambda_all);
        obj_new = (1-lambda_all)*c + lambda_all*d_omega;
        if (abs(step_ratio)<step_ratio_min)
            lambda_all = lambda_all_pre;
            break;
        end
        if (sum(log(obj_new(obj_new~=0)))>sum(log(obj(obj~=0)))) 
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update mu, i.e. the means of the Gaussian mixture components %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu_all_out = mu_all_pre;
mu_out = mu_pre;

for (ii=1:L)
ite = 1;
d_mu_tmp = zeros(N,T,L);
d_omega_all = zeros(N,T,L);
for (i=1:L)
    d_mu_tmp(:,:,i)=sqrt(abs(Rvar./(phi_all_pre(i)+Rvar)));
    d_omega_all(:,:,i) = d_mu_tmp(:,:,i).*exp(-0.5./(phi_all_pre(i)+Rvar).*(mu_all_pre(i)-Rhat).^2)*omega_all_pre(i);
end

while(ite<=max_ite)

    ite = ite+1;
    d_omega_all(:,:,ii) = d_mu_tmp(:,:,ii).*exp(-0.5./(phi_all_pre(ii)+Rvar).*(mu_all_pre(ii)-Rhat).^2)*omega_all_pre(ii);

    mu_tmp_ii=(-(mu_all_pre(ii)-Rhat)./(phi_all_pre(ii)+Rvar));
    d_omega = sum(d_omega_all, 3);

    mu_der_first_ii = lambda_all_pre./((1-lambda_all_pre)*c+lambda_all_pre*d_omega).*(d_omega_all(:,:,ii)).*mu_tmp_ii;
    
    obj = (1-lambda_all_pre)*c + lambda_all_pre*d_omega;
    step_ratio = 0;
    if (sum(mu_der_first_ii)>0)&&(mu_all_pre(ii)>0)
        step_ratio = step_ratio_init;
    elseif (sum(mu_der_first_ii)>0)&&(mu_all_pre(ii)<0)
        step_ratio = -step_ratio_init;
    elseif (sum(mu_der_first_ii)<0)&&(mu_all_pre(ii)>0)
        step_ratio = -step_ratio_init;
    else
        step_ratio = step_ratio_init;
    end
    ite_inner = 0;
    mu_all = mu_all_pre;
    while(ite_inner<=max_ite_inner)
        ite_inner=ite_inner+1;
        mu_all(ii) = mu_all_pre(ii)*(1+step_ratio);
        d_omega_all_tmp = d_omega_all;
        d_omega_all_tmp(:,:,ii)=d_mu_tmp(:,:,ii).*exp(-0.5./(phi_all_pre(ii)+Rvar).*(mu_all(ii)-Rhat).^2)*omega_all_pre(ii);
        d_omega_tmp = sum(d_omega_all_tmp, 3);
        obj_new = (1-lambda_all_pre)*c + lambda_all_pre*d_omega_tmp;
        if (abs(step_ratio)<step_ratio_min)
            mu_all(ii) = mu_all_pre(ii);
            break;
        end
        if (sum(log(obj_new(obj_new~=0)))>sum(log(obj(obj~=0)))) 
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end
    end

    mu = zeros(N,T,L);
    for (i=1:L)
        mu(:,:,i)=repmat(mu_all(i), N, T);
    end

    if (norm(mu_all(ii)-mu_all_pre(ii), 'fro')/norm(mu_all(ii), 'fro')<tol)
        break;
    end

    mu_all_pre = mu_all;
    mu_pre = mu;
end
mu_all_out(ii) = mu_all(ii);
mu_out(:,:,ii) = mu(:,:,ii);
mu_all_pre = mu_all_pre_old;
mu_pre = mu_pre_old;
end
% Output the maximizing mu
mu_all = mu_all_out;
mu = mu_out;

% Reinstate the mu values
mu_all_pre = mu_all_pre_old;
mu_pre = mu_pre_old;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update the phi, i.e. the variances of the Gaussian mixture components %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phi_all_out = phi_all_pre;
phi_out = phi_pre;
for (ii=1:L)
ite = 1;
d_omega_all = zeros(N,T,L);
for (i=1:L)
    d_omega_all(:,:,i) = sqrt(abs(Rvar./(phi_all_pre(i)+Rvar))).*exp(-0.5./(phi_all_pre(i)+Rvar).*(mu_all_pre(i)-Rhat).^2)*omega_all_pre(i);
end

while(ite<=max_ite)

    ite = ite+1;

    d_omega_all(:,:,ii) = sqrt(abs(Rvar./(phi_all_pre(ii)+Rvar))).*exp(-0.5./(phi_all_pre(ii)+Rvar).*(mu_all_pre(ii)-Rhat).^2)*omega_all_pre(ii);

    phi_tmp_ii = -0.5./(phi_all_pre(ii)+Rvar) + 0.5./((phi_all_pre(ii)+Rvar).^2).*(mu_all_pre(ii)-Rhat).^2;
    d_omega = sum(d_omega_all, 3);

    phi_der_first_ii = zeros(N,T);
    phi_der_first_ii = lambda_all_pre./((1-lambda_all_pre)*c+lambda_all_pre*d_omega).*(d_omega_all(:,:,ii)).*phi_tmp_ii;

    obj=(1-lambda_all_pre)*c + lambda_all_pre*d_omega;
    step_ratio = 0;
    if (sum(phi_der_first_ii)>0)
        step_ratio = step_ratio_init;
    else
        step_ratio = -step_ratio_init;
    end
    ite_inner = 0;
    phi_all = phi_all_pre;
    while(ite_inner<=max_ite_inner)
        ite_inner = ite_inner + 1;
        phi_all(ii) = phi_all_pre(ii)*(1+step_ratio);
        phi_all = max(phi_all_min, phi_all);  % minimum variance
        d_omega_all_tmp = d_omega_all;
        d_omega_all_tmp(:,:,ii) = sqrt(abs(Rvar./(phi_all(ii)+Rvar))).*exp(-0.5./(phi_all(ii)+Rvar).*(mu_all_pre(ii)-Rhat).^2)*omega_all_pre(ii);
        d_omega_tmp = sum(d_omega_all_tmp, 3);
        obj_new = (1-lambda_all_pre)*c + lambda_all_pre*d_omega_tmp;
        if(any(isinf(obj_new)))
            any(isinf(obj_new))
        end
        if (abs(step_ratio)<step_ratio_min)
            phi_all(ii) = phi_all_pre(ii);
            break;
        end
        if (sum(log(obj_new(obj_new~=0)))>sum(log(obj(obj~=0)))) 
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end
    end

    phi_all = max(phi_all_min, phi_all);
    phi = zeros(N,T,L);
    for (i=1:L)
        phi(:,:,i)=repmat(phi_all(i),N,T);
    end

    if (norm(phi_all(ii)-phi_all_pre(ii), 'fro')/norm(phi_all(ii), 'fro')<tol)
        break;
    end

    phi_all_pre = phi_all;
    phi_pre = phi;
end
phi_all_out(ii) = phi_all(ii);
phi_out(:,:,ii) = phi(:,:,ii);

phi_all_pre = phi_all_pre_old;
phi_pre = phi_pre_old;
end
% Output the maximizing phi
phi_all = phi_all_out;
phi = phi_out;

% Reinstate the phi values
phi_all_pre = phi_all_pre_old;
phi_pre = phi_pre_old;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update the omega, the Gaussian mixture weights %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_out = alpha_pre;
alpha_all_out = alpha_all_pre;

d = zeros(N,T,L);
for (i=1:L)
    d(:,:,i) = sqrt(abs(Rvar./(phi_all_pre(i)+Rvar))).*exp(-0.5./(phi_all_pre(i)+Rvar).*(mu_all_pre(i)-Rhat).^2);
end

for (ii=1:L)
ite = 1;
% once one omega changes, all omega would change
while(ite<=max_ite)

    ite = ite+1;
    d_omega = zeros(N,T);
    for (i=1:L)
        d_omega = d_omega + d(:,:,i)*omega_all_pre(i);
    end

    alpha_der_first_ii = zeros(N,T, L);
    for (i=1:L)
        if (i==ii)
            alpha_der_first_ii(:,:,i) = lambda_all_pre*d(:,:,ii)./((1-lambda_all_pre)*c+lambda_all_pre*d_omega)*(omega_all_pre(ii)-omega_all_pre(ii)^2);
        else
            alpha_der_first_ii(:,:,i) = lambda_all_pre*d(:,:,ii)./((1-lambda_all_pre)*c+lambda_all_pre*d_omega)*(-omega_all_pre(ii)*omega_all_pre(i));
        end
    end
    alpha_der_first_ii = sum(alpha_der_first_ii,3);

    obj=(1-lambda_all_pre)*c + lambda_all_pre*d_omega;
    step_ratio=0;
    if (sum(alpha_der_first_ii)>0)&&(alpha_all_pre(ii)>0)
        step_ratio = step_ratio_init;
    elseif (sum(alpha_der_first_ii)>0)&&(alpha_all_pre(ii)<0)
        step_ratio = -step_ratio_init;
    elseif (sum(alpha_der_first_ii)<0)&&(alpha_all_pre(ii)>0)
        step_ratio = -step_ratio_init;
    else
        step_ratio = step_ratio_init;
    end

    ite_inner = 0;
    alpha_all = alpha_all_pre;
    while(ite_inner<max_ite_inner)
        ite_inner = ite_inner + 1;
        alpha_all(ii) = alpha_all_pre(ii)*(1+step_ratio);
        alpha_all_exp = exp(alpha_all);
        omega_all = alpha_all_exp/sum(alpha_all_exp);
        d_omega_tmp = zeros(N,T);
        for (i=1:L)
            d_omega_tmp = d_omega_tmp + d(:,:,i)*omega_all(i);
        end
        obj_new = (1-lambda_all_pre)*c + lambda_all_pre*d_omega_tmp;
        if (abs(step_ratio)<step_ratio_min)
            alpha_all(ii) = alpha_all_pre(ii);
            break;
        end
        if (sum(log(obj_new(obj_new~=0)))>sum(log(obj(obj~=0))))
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
%    fprintf('%5.5f   %5.5f   %5.5f\n', omega_all(i), mu_all(i), phi_all(i))
%end

end
