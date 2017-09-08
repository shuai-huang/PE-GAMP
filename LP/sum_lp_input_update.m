%
% MAP parameter estimation of the PE-GAMP for Laplace prior input channel
%
% The simplified PE-GAMP where the parameters at different variable nodes are the same
%
% max_ite, max_ite_inner, step_ratio_init should be set appropriately
%
% lambda - Laplace distribution prior (0, infty))
%
% Shuai Huang, The Johns Hopkins University.
% E-mail: shuang40@jhu.edu
% Date: 09/03/2017
% 


function [lambda, lambda_all] = sum_lp_input_update(Rhat, Rvar, lambda_pre, lambda_all_pre)

max_ite = 100;              % maximum number of iterations
tol = 1e-6;                 % convergence criteiron
max_ite_inner = 1000;       % maximum number of iterations in the inner loop
step_ratio_init = 0.01;     % initial step size, set to a relatively small number for stability
dec_rate = 0.9;             % the decreasing rate of the step ratio
step_ratio_min = 1e-6;      % the minimum step size

% The minimum lambda parameter value
lambda_min = 1e-12;         % mimimum lambda value


% Calcualte Problem dimensions
[N, T] = size(Rhat);

ite = 1;
while(ite<=max_ite)
    ite = ite+1;

    block_1 = sqrt(pi*Rvar/2).*erfcx(-(Rhat-lambda_pre.*Rvar)./sqrt(2*Rvar));
    block_2 = sqrt(pi*Rvar/2).*erfcx((Rhat+lambda_pre.*Rvar)./sqrt(2*Rvar));

    sig = sqrt(Rvar);                           % Gaussian prod std dev
    muL = Rhat + lambda_pre.*Rvar;              % Lower integral mean
    muU = Rhat - lambda_pre.*Rvar;              % Upper integral mean
    muL_over_sig = muL ./ sig;
    muU_over_sig = muU ./ sig;
    cdfL = normcdf(-muL_over_sig);              % Lower cdf
    cdfU = normcdf(muU_over_sig);               % Upper cdf
    cdfRatio = cdfL ./ cdfU;                    % Ratio of lower-to-upper CDFs
    SpecialConstant = exp( (muL.^2 - muU.^2) ./ (2*Rvar) ) .* cdfRatio;
    NaN_Idx = isnan(SpecialConstant);           % Indices of trouble constants

    % For the "trouble" constants (those with muL's and muU's
    % that are too large to give accurate numerical answers),
    % we will effectively peg the special constant to be Inf or
    % 0 based on whether muL dominates muU or vice-versa
    SpecialConstant(NaN_Idx & (-muL >= muU)) = Inf;
    SpecialConstant(NaN_Idx & (-muL < muU)) = 0;

    % Compute the ratio normpdf(a)/normcdf(a) for
    % appropriate upper- and lower-integral constants, a
    RatioL = 2/sqrt(2*pi) ./ erfcx(muL_over_sig / sqrt(2));
    RatioU = 2/sqrt(2*pi) ./ erfcx(-muU_over_sig / sqrt(2));

    % Now compute the first posterior moment...
    der_first = 1./lambda_pre + (1 ./ (1 + SpecialConstant.^(-1))) .* (muL - sig.*RatioL) + (1 ./ (1 + SpecialConstant)) .* (-muU - sig.*RatioU);

    obj_old = lambda_pre.*(block_1+block_2);    % old objective function value
    step_ratio = 0;
    if (sum(der_first)>0)
        step_ratio = step_ratio_init;
    else
        step_ratio = -step_ratio_init;
    end
    ite_inner = 0; 
    while(ite_inner<=max_ite_inner)
        ite_inner=ite_inner+1;
        lambda_all = lambda_all_pre*(1+step_ratio);
        tmp_block_1 = sqrt(pi*Rvar/2).*erfcx(-(Rhat-lambda_all.*Rvar)./sqrt(2*Rvar));
        tmp_block_2 = sqrt(pi*Rvar/2).*erfcx((Rhat+lambda_all.*Rvar)./sqrt(2*Rvar));

        obj_new = lambda_all.*(tmp_block_1+tmp_block_2);    % new objective function value
        if (abs(step_ratio)<step_ratio_min)
            lambda_all = lambda_all_pre;
            break;
        end

        obj_ratio = obj_old./obj_new;
        if (any(isnan(obj_ratio)))
            idx_nan = 1:N;
            idx_nan = idx_nan(isnan(obj_ratio));
            for (idxn = idx_nan)
                Rhat_nan = Rhat(idxn); Rvar_nan = Rvar(idxn); sig_nan = sqrt(Rvar_nan);
                mu_pre_nan = [(Rhat_nan-lambda_all_pre*Rvar_nan)/sig_nan (Rhat_nan+lambda_all_pre*Rvar_nan)/sig_nan];
                mu_nan = [(Rhat_nan-lambda_all*Rvar_nan)/sig_nan (Rhat_nan+lambda_all*Rvar_nan)/sig_nan];
                [Um, Ui] = min([-mu_pre_nan(1) mu_pre_nan(2) -mu_nan(1) mu_nan(2)]);
                mu_pre_nan_sq = mu_pre_nan.^2;
                mu_nan_sq = mu_nan.^2;
                mu_pre_nan_new_sq = mu_pre_nan_sq-Um^2;
                mu_nan_new_sq = mu_nan_sq-Um^2;
                obj_nan = (exp(0.5*(mu_pre_nan_new_sq(1)))*normcdf(mu_pre_nan(1)) + exp(0.5*(mu_pre_nan_new_sq(2)))*normcdf(-mu_pre_nan(2)))/( exp(0.5*(mu_nan_new_sq(1)))*normcdf(mu_nan(1)) + exp(0.5*(mu_nan_new_sq(2)))*normcdf(-mu_nan(2)) );
                obj_nan = obj_nan*lambda_all_pre/lambda_all;
                if (isnan(obj_nan))
                    obj_ratio(idxn) = lambda_all_pre/lambda_all;
                else
                    obj_ratio(idxn) = obj_nan;
                end
            end
        end

        if (sum(log(obj_ratio))<0)
            break;
        else
            step_ratio=step_ratio*dec_rate;
        end 

    end


    lambda_all = max(lambda_min, lambda_all);
    lambda = repmat(lambda_all, N, T);% - der_first/(-sum(der_second));

    if (norm(lambda_all-lambda_all_pre, 'fro')/norm(lambda_all, 'fro') < tol)
        break;
    end

    lambda_all_pre = lambda_all;
    lambda_pre = lambda;
end

%fprintf('%5.5f\n', lambda_all)

end
