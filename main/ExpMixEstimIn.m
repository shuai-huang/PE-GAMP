classdef ExpMixEstimIn < EstimIn
    % ExpMixEstimIn:  Inputs a soft thresholding scalar input function.
    % Allows GAMP to be used to solve "min_x 1/2/var*norm(y-A*x,2)^2 + lambda*norm(x,1)".
    
    properties
        omega = 1;  % Weights
        %lambda = the gain on the ell1 term in the MAP cost.
        %The soft threshold is set according to the expression thresh = lambda * mur;
        lambda;
        maxSumVal = true;   % Max-sum GAMP (true) or sum-product GAMP (false)?
        autoTune = false;   % Perform tuning of lambda (true) or not (false)?
        disableTune = false;% Set to true to temporarily disable tuning
        tuneDim = 'joint';  % Parameter tuning across rows and columns ('joint'),
        % or just columns ('col') or rows ('row')
        counter = 0;        % Counter to delay tuning
        sureParams = ...
            struct('method',1,...
            'damp',1,...
            'decayDamp',1,...
            'dampFac',.95,...
            'delay',0,...
            'step',10,... 
            'gm_minvar',0,...
            'bgm_alg',0,...
            'gm_step',1,...
            'GM',[],...
            'initVar',[],...
            'initSpar',[]);       
        % sureParams:
            % method: selects the method to minimize SURE
                % (1) a bisection search method on Gaussian Mixture SURE
                % (2) gradient descent on Gaussian Mixture SURE
                % (3) approximate gradient descent 
            % damp: stepsize between new and old lambda (1 = no amount
            % of damping)
            % decayDamp: decrease damping parameter (stepsize) by 1-dampFac
            % damp_fac: factor to reduce damp by
            % delay: number of gamp iterations between lambda tuning 
            % step: initial step size in approximate gradient descent 
            % gm_minvar: set to 1 to force minimum GM component variance of
            % rvar
            % gm_alg: set to 1 to learn Bernoulli-GM on X, then convolve
            % with N(0,rvar) to obtain distribution on rhat, otherwise,
            % learn Gaussian mixture distribution without the Bernoulli
            % component
            % gm_step: initial stepsize for gradient descent on Gaussian
            % Mixture SURE
            % GM: a place to store the GM approximation to rhat 
          
    end
    
    properties (Hidden)
        mixWeight = 1;     % Mixture weight (used for EM tuning w/ SparseScaEstimIn)
        lam_left = [];     
        lam_right = [];
        tune_it = 0;
    end
    
    methods
        % Constructor
        function obj = ExpMixEstimIn(omega, lambda, maxSumVal, varargin)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.omega = omega;
                obj.lambda = lambda;
                if nargin >= 3 && ~isempty(maxSumVal) && isscalar(maxSumVal)
                    obj.maxSumVal = logical(maxSumVal);
                    
                    if nargin >= 4
                        for i = 1:2:length(varargin)
                            obj.(varargin{i}) = varargin{i+1};
                        end
                    end
                end
            end
        end

        function obj = set.omega(obj,omega)
            obj.omega = omega;
        end

        function obj = set.lambda(obj, lambda)
            obj.lambda = lambda;
        end
        
        function set.disableTune(obj, flag)
            assert(isscalar(flag), ...
                'ExpMixEstimIn: disableTune must be a logical scalar');
            obj.disableTune = flag;
        end
        
        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = 0;
            if obj.maxSumVal
                lambda_ave = mean(squeeze(obj.lambda(1,1,:)));
                %var0 = 5e-4; % a reasonable constant?
                var0 = 2./(lambda_ave.^2);
                valInit = -inf; % should fix this...
            else
                lambda_ave = mean(squeeze(obj.lambda(1,1,:)));
                var0 = 2./(lambda_ave.^2);
                valInit = 0; % should fix this...
            end
        end
        
        % Carry out soft thresholding
        function [xhat,xvar,val] = estim(obj, rhat, rvar)

            %Get the number of mixture components
            L = size(obj.omega, 3);

            %Grab the signal dimension
            [N, T] = size(rhat);

            %Expand scalar estimator if needed
            omega = resize(obj.omega, N, T, L);
            lambda = resize(obj.lambda, N, T, L);

            if ~obj.maxSumVal
                % Compute sum-product GAMP updates
                
                % To avoid numerical problems (0/0) when evaluating
                % ratios of Gaussian CDFs, impose a firm cap on the
                % maximum value of entries of rvar
                %rvar = min(rvar, 700);

                xhat_mix_numerator = zeros(N,T,L);
                SecondMoment_mix_numerator = zeros(N,T,L);
                mix_denominator = zeros(N,T,L);

                OLC = zeros(N,T,L);

                sig = sqrt(rvar);                           % Gaussian prod std dev
                %muL_over_sig_all = zeros(N,T,L);
                muU_over_sig_all = zeros(N,T,L);
                SC_U_all = zeros(N,T,L);
                %SC_L_all = zeros(N,T,L);
                %muL_all = zeros(N,T,L);
                muU_all = zeros(N,T,L);

                for (idx_mix = 1:L)
                lambda_mix = lambda(:,:,idx_mix);
                omega_mix = omega(:,:,idx_mix);
                omega_lambda_mix = omega_mix.*lambda_mix;
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                %muL = rhat + lambda_mix.*rvar;          	% Lower integral mean
                muU = rhat - lambda_mix.*rvar;          	% Upper integral mean
                %muL_all(:,:,idx_mix) = muL;
                muU_all(:,:,idx_mix) = muU;
                %muL_over_sig = muL ./ sig;
                muU_over_sig = muU ./ sig;
                %muL_over_sig_all(:,:,idx_mix) = muL_over_sig;
                muU_over_sig_all(:,:,idx_mix) = muU_over_sig;

                C_U = erfcx(-muU_over_sig / sqrt(2));
                %C_L = erfcx(muL_over_sig / sqrt(2));

                %OLC(:,:,idx_mix) = omega_lambda_mix.*(C_U+C_L);
                OLC(:,:,idx_mix) = omega_lambda_mix.*(C_U);
                
                SC_U = sqrt(pi/2)*C_U;
                %SC_L = sqrt(pi/2)*C_L;
                SC_U_all(:,:,idx_mix) = SC_U;
                %SC_L_all(:,:,idx_mix) = SC_L;

                xhat_mix_numerator(:,:,idx_mix) = omega_lambda_mix.*(sig+muU.*SC_U);
                SecondMoment_mix_numerator(:,:,idx_mix) = omega_lambda_mix.*(sig.*muU + (rvar+muU.^2).*SC_U );
                mix_denominator(:,:,idx_mix) = omega_lambda_mix.*(SC_U);
                
                end

                xhat_numerator = sum(xhat_mix_numerator, 3);
                SecondMoment_numerator = sum(SecondMoment_mix_numerator, 3);
                denominator = sum(mix_denominator, 3);

                xhat = xhat_numerator ./ denominator;
                SecondMoment = SecondMoment_numerator ./ denominator;
                %if (any(isnan(xhat)))
                %    nan_idx=1:N;
                %    nan_idx=nan_idx(isnan(xhat));
                %    fprintf('%5d   %5d   %5d\n', muU(nan_idx(1)), SC_U(nan_idx(1)), rvar(nan_idx(1)));
                %    fprintf('%5d\n', SecondMoment(nan_idx(1)))
                %end

                if (any(isnan(xhat)))
                    nan_idx=1:N;
                    nan_idx=nan_idx(isnan(xhat));
                    for (idxn=nan_idx)
                        SC_U_all_nan = SC_U_all(idxn,:,:);
                        %SC_L_all_nan = SC_L_all(idxn,:,:);
                        %if (any(isinf(SC_U_all_nan)))
                            [Um, Ui]=max(muU_over_sig_all(idxn,:,:));
                            xhat(idxn) = muU_all(idxn,:,Ui(1));
                            SecondMoment(idxn) = rvar(idxn)+muU_all(idxn,:,Ui(1))^2;
                        %else
                            %[Lm, Li]=min(muL_over_sig_all(idxn,:,:));
                            %xhat(idxn) = muL_all(idxn,:,Li(1));
                            %SecondMoment(idxn) = rvar(idxn)+muL_all(idxn,:,Li(1))^2;
                        %end
                    end
                end

                if (any(isnan(SecondMoment)))
                    nan_idx=1:N;
                    nan_idx=nan_idx(isnan(SecondMoment));
                    for (idxn=nan_idx)
                        SC_U_all_nan = SC_U_all(idxn,:,:);
                        %SC_L_all_nan = SC_L_all(idxn,:,:);
                        %if (any(isinf(SC_U_all_nan)))
                            [Um, Ui]=max(muU_over_sig_all(idxn,:,:));
                            xhat(idxn) = muU_all(idxn,:,Ui(1));
                            SecondMoment(idxn) = rvar(idxn)+muU_all(idxn,:,Ui(1))^2;
                        %else
                            %[Lm, Li]=min(muL_over_sig_all(idxn,:,:));
                            %xhat(idxn) = muL_all(idxn,:,Li(1));
                            %SecondMoment(idxn) = rvar(idxn)+muL_all(idxn,:,Li(1))^2;
                        %end
                    end
                end
                %any(isnan(SecondMoment))

                if (any(isinf(xhat)))
                    inf_idx=1:N;
                    inf_idx=inf_idx(isinf(xhat));
                    for (idxn=inf_idx)
                        SC_U_all_inf = SC_U_all(idxn,:,:);
                        %SC_L_all_inf = SC_L_all(idxn,:,:);
                        %if (any(isinf(SC_U_all_inf)))
                            [Um, Ui]=max(muU_over_sig_all(idxn,:,:));
                            xhat(idxn) = muU_all(idxn,:,Ui(1));
                            SecondMoment(idxn) = rvar(idxn)+muU_all(idxn,:,Ui(1))^2;
                        %else
                            %[Lm, Li]=min(muL_over_sig_all(idxn,:,:));
                            %xhat(idxn) = muL_all(idxn,:,Li(1));
                            %SecondMoment(idxn) = rvar(idxn)+muL_all(idxn,:,Li(1))^2;
                        %end
                    end  
                end

                if (any(isinf(SecondMoment)))
                    inf_idx=1:N;
                    inf_idx=inf_idx(isinf(SecondMoment));
                    for (idxn=inf_idx)
                        SC_U_all_inf = SC_U_all(idxn,:,:);
                        %SC_L_all_inf = SC_L_all(idxn,:,:);
                        %if (any(isinf(SC_U_all_inf)))
                            [Um, Ui]=max(muU_over_sig_all(idxn,:,:));
                            xhat(idxn) = muU_all(idxn,:,Ui(1));
                            SecondMoment(idxn) = rvar(idxn)+muU_all(idxn,:,Ui(1))^2;
                        %else
                            %[Lm, Li]=min(muL_over_sig_all(idxn,:,:));
                            %xhat(idxn) = muL_all(idxn,:,Li(1));
                            %SecondMoment(idxn) = rvar(idxn)+muL_all(idxn,:,Li(1))^2;
                        %end
                    end  
                end  

                %fprintf('%5d   %5d   %5d\n', length(rhat(isnan(rhat))), length(rvar(isnan(rvar))), length(xhat(isnan(xhat))))

                if (any(isnan(xhat)))
                    nan_idx=1:N;
                    nan_idx=nan_idx(isnan(xhat));
                    fprintf('%5d   %5d   %5d   %5d\n', muU(nan_idx(1)), SC_U(nan_idx(1)), rhat(nan_idx(1)), rvar(nan_idx(1)));
                    fprintf('%5d   %5d\n', SecondMoment(nan_idx(1)), length(rhat(isnan(rhat))))
                end

                xvar = SecondMoment - xhat.^2;
                %fprintf('%5d   %5d   %5d\n', xvar(isinf(xvar)), SecondMoment(isinf(xvar)), xhat(isinf(xvar)));
                %if (any(isnan(xvar)))
                %    fprintf('%5d   %5d   %5d\n', length(xhat(isnan(xhat))), length(SecondMoment(isnan(SecondMoment))), length(xvar(isnan(xvar))))
                %    nan_idx=1:N;
                %    nan_idx=nan_idx(isnan(xvar));
                %    fprintf('%5d   %5d\n', xhat(nan_idx(1)), SecondMoment(nan_idx(1)))
                %    rvar(nan_idx(1))
                %    squeeze(muU_all(nan_idx(1),:,:))
                %    squeeze(muL_all(nan_idx(1),:,:))
                %end
                % *********************************************************
                
                % Perform EM parameter tuning, if desired
                if obj.autoTune && ~obj.disableTune
                    
                    if (obj.counter>0), % don't tune yet
                        obj.counter = obj.counter-1; % decrement counter
                    else % tune now
                        
                        % Start by computing E[|x_n| | y]...
                        mu = (1 ./ (1 + SpecialConstant)) .* (muU + sig.*RatioU) ...
                            - (1 ./ (1 + SpecialConstant.^(-1))) .* ...
                            (muL - sig.*RatioL);
                        [N, T] = size(xhat);
                        
                        %Average over all elements, per column, or per row
                        switch obj.tuneDim
                            case 'joint'
                                lambda = N*T / sum(obj.mixWeight(:).*mu(:));
                            case 'col'
                                lambda = repmat(N ./ sum(obj.mixWeight.*mu), ...
                                    [N 1]);
                            case 'row'
                                lambda = repmat(T ./ sum(obj.mixWeight ...
                                    .*mu, 2), [1 T]);
                            otherwise
                                error('Invalid tuning dimension in ExpMixEstimIn'); 
                        end
                        
                        if any( lambda(:) <= 0 | isnan(lambda(:)) )
                            warning('EM update of lambda was negative or NaN...ignoring')
                            lambda = obj.lambda;
                        end
                        obj.lambda = lambda;
                        
                    end
                end
                
                % Lastly, compute negative KL divergence:
                % \int_x p(x|y) log(p(x)/p(x|y)) dx
                
                %                 % Old way of computing.  It handles difficult cases
                %                 % incorrectly.
                %                 NormConL = obj.lambda/2 .* ...              % Mass of lower integral
                %                     exp( (muL.^2 - rhat.^2) ./ (2*rvar) ) .* cdfL;
                %                 NormConU = obj.lambda/2 .* ...              % Mass of upper integral
                %                     exp( (muU.^2 - rhat.^2) ./ (2*rvar) ) .* cdfU;
                %                 NormCon = NormConL + NormConU;      % Posterior normaliz. constant recip.
                %                 NormCon(isnan(NormCon)) = 1;        % Not much we can do for trouble ones
                %                 NormCon(NormCon == Inf) = 1;
                
                if (nargout >= 3)
                    % Calculate the log scale factor
                    logNormCon = log(1/2) - rhat.^2./(2*rvar) + log(sum(OLC,3));
                    
                    val = logNormCon + ...
                        0.5*(log(2*pi*rvar) + ...
                        (xvar + (xhat - rhat).^2)./rvar);
                    
                    %Old fix, no longer needed
                    %val(val == -Inf) = -1e4;    % log(NormCon) can == -Inf
                end
                
            else %if obj.maxSumVal
                
                % tune lambda to minimize the SURE of the estimator's MSE.
                % more details are in "Sparse multinomial logistic 
                
                if obj.autoTune && ~obj.disableTune
                    
                    debug = 0; % applies to all modes, produces diagnostic plots at every GAMP iteration

                    if (obj.counter>0), % don't tune yet
                        obj.counter = obj.counter-1; % decrement counter
                    else % tune now
                        
                        obj.tune_it = obj.tune_it+1;

                        [N,T] = size(rhat);
                        
                        switch obj.tuneDim
                            case 'joint'
                                dim = 1;
                                % assume uniform variance
                                c = mean(rvar(:));                     
                            case 'col'
                                dim = T;
                                % assume uniform variance
                                c = mean(rvar,1);
                            case 'row'
                                dim = N;
                                % assume uniform variance
                                c = mean(rvar,2);
                        end
                        if length(obj.sureParams.damp) ~= dim
                            obj.sureParams.damp = obj.sureParams.damp(1)*ones(dim,1);
                        end
                        if length(obj.sureParams.step) ~= dim
                            obj.sureParams.step = obj.sureParams.step(1)*ones(dim,1);
                        end
                        if length(obj.sureParams.gm_step) ~= dim
                            obj.sureParams.gm_step = obj.sureParams.gm_step(1)*ones(dim,1);
                        end
                        if size(obj.sureParams.GM,1) ~= dim
                            obj.sureParams.GM = cell(dim, 1);
                        end
                        
                        % loop over tuning dimensions (joint, row or
                        % column)
                        for t = 1:dim
                            
                            % lam_max is the smallest value of lambda which
                            % will set every x to zero
                            % lam_min is the smallest value of lambda which
                            % empricial SURE is still valid, GM sure may be
                            % different
                            switch obj.tuneDim
                                case 'joint'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda);
                                    % compute lambda max
                                    lam_max = min(max(abs(rhat)./c));
                                    lam_min = max(min(abs(rhat)./c));
                                    idx = 1:(N*T);
                                case 'col'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda(:,t));
                                    % compute lambda max
                                    lam_max = max(abs(rhat(:,t))./c(t));
                                    lam_min = min(abs(rhat(:,t))./c(t));
                                    idx = (1:N) + (t-1)*N;
                                case 'row'
                                    % find initial lambda
                                    lambda0 = mean(obj.lambda(t,:));
                                    % compute lambda max
                                    lam_max = max(abs(rhat(t,:))./c(t));
                                    lam_min = min(abs(rhat(t,:))./c(t));
                                    idx = (1:N:(N*T)) + (t-1);
                            end
                            
                            % select method to optimize SURE
                            switch obj.sureParams.method
                                case 1 % Gaussian mixture with bisection search on gradient
                                    lambda = obj.minSureGMbisect(lambda0, rhat(idx), rvar(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                case 2 % Gaussian mixture with gradient descent
                                    lambda = obj.minSureGMgrad(lambda0, rhat(idx), rvar(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                case 3 % approximate gradient descent
                                    lambda = obj.minSureGrad(lambda0, rhat(idx), c(t), lam_max, lam_min, t, obj.tune_it, debug);
                                    
                            end
                            
                            % implement damping on lambda
                            damp = obj.sureParams.damp(t); % damping on lambda update 
                            lambda = 10^(damp * log10(lambda) + (1-damp)*log10(lambda0));

                            % set new lambda
                            switch obj.tuneDim
                                case 'joint'
                                    obj.lambda = lambda;
                                case 'col'
                                    if numel(obj.lambda) == 1
                                        obj.lambda = obj.lambda * ones(N,T);
                                    end
                                    obj.lambda(:,t) = lambda * ones(N,1);
                                case 'row'
                                    if numel(obj.lambda) == 1
                                        obj.lambda = obj.lambda * ones(N,T);
                                    end
                                    obj.lambda(t,:) = lambda * ones(1,T);
                            end
                            
                            if obj.sureParams.decayDamp 
                                obj.sureParams.damp(t) = obj.sureParams.damp(t) * obj.sureParams.dampFac; 
                            end
                        end
                        
                        obj.counter = obj.sureParams.delay;
                        
                    end
                    
                end
                
                % Compute max-sum GAMP updates
                
                %Compute the thresh
                ite_mix = 0;
                xhat_pre = rhat;
                while(ite_mix<=100)
                    ite_mix = ite_mix+1;
                    lambda_new_numerator = zeros(N,T,L);
                    lambda_new_denominator = zeros(N,T,L);
                    for (idx_mix=1:L)
                        lambda_mix = lambda(:,:,idx_mix);
                        omega_mix = omega(:,:,idx_mix);
                        lambda_new_denominator(:,:,idx_mix)=omega_mix.*lambda_mix.*exp(-lambda_mix.*abs(xhat_pre))+1e-16;
                        lambda_new_numerator(:,:,idx_mix) = lambda_new_denominator(:,:,idx_mix).*lambda_mix;
                    end
                    lambda_new = sum(lambda_new_numerator,3)./sum(lambda_new_denominator,3);
                    thresh = lambda_new.*rvar;
                    xhat = max(0,abs(rhat)-thresh) .* sign(rhat);
                    if (max(abs((xhat-xhat_pre)./xhat))<1e-6)
                        break;
                    end
                    xhat_pre = xhat;
                end
                
                %Estimate the variance
                %xvar = rvar .* (mean(double(abs(xhat) > 0))*ones(size(xhat)));
                %xvar = rvar .* (abs(xhat) > 0);
                s_numerator = zeros(N,T,L);
                s2_numerator = zeros(N,T,L);
                s_denominator = zeros(N,T,L);
                for (idx_mix=1:L)
                    lambda_mix = lambda(:,:,idx_mix);
                    omega_mix = omega(:,:,idx_mix);
                    s_denominator(:,:,idx_mix)=omega_mix.*lambda_mix.*exp(-lambda_mix.*abs(xhat))+1e-16;
                    s_numerator(:,:,idx_mix) = s_denominator(:,:,idx_mix).*lambda_mix;
                    s2_numerator(:,:,idx_mix) = s_denominator(:,:,idx_mix).*(lambda_mix.^2);
                end
                s_mom = (sum(s2_numerator,3).*sum(s_denominator,3) -  sum(s_numerator,3).^2 )./(sum(s_denominator,3) ).^2;
                s_mom(isnan(s_mom))=0;
                s_mom(abs(xhat)==0)=Inf;

                xvar = rvar./(repmat(1,N,T)-rvar.*s_mom);

                if (nargout >= 3)
                    %Output negative cost
                    %val = -1*obj.lambda*abs(rhat);
                    %val = -1*obj.lambda(1).*abs(xhat);	% seems to work better
                    val = log(sum(s_denominator,3));
                end
            end
        end
        
        % Computes p(y) for y = x + v, with x ~ p(x), v ~ N(0,yvar)
        function py = plikey(obj, y, yvar)
            rhat = y;
            rvar = yvar;
            %Get the number of mixture components
            L = size(obj.omega, 3);

            %Grab the signal dimension
            [N, T] = size(rhat);

            %Expand scalar estimator if needed
            omega = resize(obj.omega, N, T, L);
            lambda = resize(obj.lambda, N, T, L);

            xhat_mix_numerator = zeros(N,T,L);
            SecondMoment_mix_numerator = zeros(N,T,L);
            mix_denominator = zeros(N,T,L);

            OLC = zeros(N,T,L);

            sig = sqrt(rvar);                           % Gaussian prod std dev
            %muL_over_sig_all = zeros(N,T,L);
            muU_over_sig_all = zeros(N,T,L);
            SC_U_all = zeros(N,T,L);
            %SC_L_all = zeros(N,T,L);
            %muL_all = zeros(N,T,L);
            muU_all = zeros(N,T,L);

            for (idx_mix = 1:L) 
                lambda_mix = lambda(:,:,idx_mix);
                omega_mix = omega(:,:,idx_mix);
                omega_lambda_mix = omega_mix.*lambda_mix;
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                %muL = rhat + lambda_mix.*rvar;             % Lower integral mean
                muU = rhat - lambda_mix.*rvar;              % Upper integral mean
                muU_all(:,:,idx_mix) = muU; 
                muU_over_sig = muU ./ sig; 
                muU_over_sig_all(:,:,idx_mix) = muU_over_sig;

                C_U = erfcx(-muU_over_sig / sqrt(2));

                OLC(:,:,idx_mix) = omega_lambda_mix.*(C_U);

            end

            OLC_sum = sum(OLC, 3);
            OLC_sum(OLC_sum==Inf) = realmax;
            py = 0.5 * exp(-(y.^2) ./ (2*yvar)) .* OLC_sum;

        end
        
        function logpy = loglikey(obj, y, yvar)
            rhat = y;
            rvar = yvar;
            %Get the number of mixture components
            L = size(obj.omega, 3);

            %Grab the signal dimension
            [N, T] = size(rhat);

            %Expand scalar estimator if needed
            omega = resize(obj.omega, N, T, L);
            lambda = resize(obj.lambda, N, T, L);


            xhat_mix_numerator = zeros(N,T,L);
            SecondMoment_mix_numerator = zeros(N,T,L);
            mix_denominator = zeros(N,T,L);

            OLC = zeros(N,T,L);

            sig = sqrt(rvar);                           % Gaussian prod std dev
            %muL_over_sig_all = zeros(N,T,L);
            muU_over_sig_all = zeros(N,T,L);
            SC_U_all = zeros(N,T,L);
            %SC_L_all = zeros(N,T,L);
            %muL_all = zeros(N,T,L);
            muU_all = zeros(N,T,L);

            for (idx_mix = 1:L) 
                lambda_mix = lambda(:,:,idx_mix);
                omega_mix = omega(:,:,idx_mix);
                omega_lambda_mix = omega_mix.*lambda_mix;
                % *********************************************************
                % Begin by computing various constants on which the
                % posterior mean and variance depend
                %muL = rhat + lambda_mix.*rvar;             % Lower integral mean
                muU = rhat - lambda_mix.*rvar;              % Upper integral mean
                muU_all(:,:,idx_mix) = muU; 
                muU_over_sig = muU ./ sig; 
                muU_over_sig_all(:,:,idx_mix) = muU_over_sig;

                C_U = erfcx(-muU_over_sig / sqrt(2));

                OLC(:,:,idx_mix) = omega_lambda_mix.*(C_U);

            end

            OLC_sum = sum(OLC, 3);
            OLC_sum(OLC_sum==Inf) = realmax;
            logpy = log( 0.5 ) + (-(y.^2) ./ (2*yvar)) + log(OLC_sum);

        end
        
        % Bisection Search method to minimize GM-SURE function
        function lambda = minSureGMbisect(obj, lambda0, rhat, rvar, c, lam_max, lam_min, t, tune_it, debug)
            
            num_it = 5; % number of bisections
            lambda_hist = nan(num_it+1,1);
            
            % learn gaussian mixture
            if obj.sureParams.bgm_alg
                L = 4; % number of GM components (after conv. with rvar)
                gm = obj.embgm(rhat,rvar,c,L,t);
            else
                L = 3; % number of GM components
                gm = obj.emgm(rhat,c,L,t);
            end

            % we assume the GM SURE function's gradient begins negative and
            % has only one root. Thus, if the gradient at lambda_max is
            % negative, we skip the bisection method and simply set lambda
            % = lambda_max.
            
            gradient = obj.gmSUREgrad(lam_max,c,gm);
            
            if gradient < 0
                % minimum not contained in "effective" lambda, terminate
                lambda = lam_max;
                lambda_hist(end) = lam_max;
            else
                % perform bisection search to find minimum
                
                % first, find initial bounds using the final bounds from 
                % the previous iteration and backtracking if necessary.
                
                % set initial search 
                if isempty(obj.lam_left)
                    obj.lam_left = lam_min;
                end
                if isempty(obj.lam_right)
                    obj.lam_right = lam_max;
                end
                
                gradient_left = obj.gmSUREgrad(obj.lam_left,c,gm);
                gradient_right = obj.gmSUREgrad(obj.lam_right,c,gm);
                
                if gradient_left <= 0 && gradient_right >= 0
                    % case 1: gradient left is negative, gradient right is
                    % positive, indicating the root is between them

                    % do nothing here...
                    
                elseif gradient_left >= 0 && gradient_right >= 0
                    % case 2: gradient left and right are positive,
                    % indicating the root is to the left
                    
                    stop = 0;
                    delta = abs(obj.lam_left - obj.lam_right);
                    scale = 1;
                    it = 1;
                    while ~stop
                        it = it + 1;
                        obj.lam_right = obj.lam_left;
                        obj.lam_left = max(obj.lam_left - scale * delta, lam_min);
                        scale = 2*scale;
                        % check gradient of new lam_right
                        gradient_left = obj.gmSUREgrad(obj.lam_left,c,gm);
                        if gradient_left < 0 || obj.lam_left == lam_min || it == 20
                            stop = 1;
                        end
                    end
                    
                elseif gradient_left <= 0 && gradient_right <= 0
                    % case 3: gradient left and right are negative,
                    % indicating the root is to the right
                    
                    stop = 0;
                    delta = abs(obj.lam_left - obj.lam_right);
                    scale = 1;
                    it = 1;
                    while ~stop
                        it = it + 1;
                        obj.lam_left = obj.lam_right;
                        obj.lam_right = min(obj.lam_right + scale * delta, lam_max);
                        scale = 2*scale;
                        % check gradient of new lam_right
                        gradient_right = obj.gmSUREgrad(obj.lam_right,c,gm);
                        if gradient_right > 0 || obj.lam_right == lam_max || it == 20
                            stop = 1;
                        end
                    end
                else
                    % in this case, our unique root assumption of GM-SURE
                    % was violated, so refrain from tuning lambda this
                    % iteration
                    lambda = min(lambda0,lam_max);
                    return;
                end

                % Bisection search 
                for it = 1:num_it;
                    
                    % bisect in log-domain
                    lambda = 10^(log10(obj.lam_left*obj.lam_right)/2);
                    lambda_hist(it) = lambda;
                    
                    % compute gradient
                    gradient = obj.gmSUREgrad(lambda,c,gm);
                    
                    if gradient < 0
                        obj.lam_left = lambda;
                    else
                        obj.lam_right = lambda;
                    end
                end
                
                % one final bisection...
                lambda = 10^(log10(obj.lam_left*obj.lam_right)/2);
                lambda_hist(end) = lambda;
                
            end

            if debug
                figure(100);clf;
                % plot samples
                semilogx(lambda_hist(1:end-1), obj.gmSUREcost(lambda_hist(1:end-1),c,gm),'bo')
                hold on
                % plot final lambda
                semilogx(lambda_hist(end), obj.gmSUREcost(lambda_hist(end),c,gm),'rs')
                % plot cost
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.gmSUREcost(lgrid,c,gm);
                semilogx(lgrid,cost,'g')
                hold off
                xlabel('lambda')
                ylabel('SURE')
                title(sprintf('minimization of objective function; gamp it = %d',tune_it))
                legend('samples','final lambda','sure cost')
                figure(101);clf;
                semilogy(lambda_hist)
                xlabel('iter')
                ylabel('lambda')
                title('lambda vs bisection iteration')
                figure(102);clf;
                histnorm(rhat,40)
                hold on
                x = linspace(.9*min(rhat), 1.1*max(rhat), 100);
                y = zeros(size(x));
                for l = 1:L
                    y = y + gm.omega(l) * normpdf(x, gm.theta(l), sqrt(gm.phi(l)));
                end
                plot(x,y,'g')
                hold off
                title('GM fit to rhat')
                legend('rhat','GM')
                drawnow;
                pause;
            end
            
        end
        
        % Gradient descent method to minimize GM SURE
        function lambda = minSureGMgrad(obj, lambda0, rhat, rvar, c, lam_max, lam_min, t, tune_it, debug)
            
            lambda0 = min(lambda0, lam_max);
            lambda = lambda0;
            
            maxit = 100;
            lambda_hist = nan(1,maxit);
            tol = 0;
            
            alpha = obj.sureParams.gm_step(t); % gradient descent initial step size
            
            % learn gaussian mixture
            if obj.sureParams.bgm_alg
                L = 4; % number of GM components
                gm = obj.embgm(rhat,rvar,c,L,t);
            else
                L = 3; % number of GM components
                gm = obj.emgm(rhat,c,L,t);
            end
            
            grad_old = 0;
            
            % apply gradient projection
            for it = 1:maxit
                lam_old = lambda;
                gradient = obj.gmSUREgrad(lambda,c,gm);
                if gradient < 0 && lambda == lam_max
                    break
                end
                lambda = max(min(lambda - alpha * gradient, lam_max),lam_min);
                
                if sign(gradient)*sign(grad_old) == 1
                    % increase step
                    stepmax = 1e6;
                    alpha = min(alpha*1.1, stepmax);
                else
                    % descrease step
                    alpha = alpha*.5;
                end
                grad_old = gradient;
                lambda_hist(it) = lambda;
                if abs(lam_old - lambda)/(lam_old+eps) < tol
                    break
                end
            end
            
            obj.sureParams.gm_step(t) = alpha;
            
            if debug
                figure(100);clf;
                % plot samples
                semilogx(lambda_hist(1:end-1), obj.gmSUREcost(lambda_hist(1:end-1),c,gm),'bo')
                hold on
                % plot final lambda
                semilogx(lambda_hist(end), obj.gmSUREcost(lambda_hist(end),c,gm),'rs')
                % plot cost
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.gmSUREcost(lgrid,c,gm);
                semilogx(lgrid,cost,'g')
                hold off
                xlabel('lambda')
                ylabel('SURE')
                title(sprintf('minimization of objective function; gamp it = %d',tune_it))
                legend('samples','final lambda','sure cost')
                figure(101);clf;
                semilogy(lambda_hist)
                xlabel('iter')
                ylabel('lambda')
                title('lambda vs iteration')
                figure(102);clf;
                histnorm(rhat,40)
                hold on
                x = linspace(.9*min(rhat), 1.1*max(rhat), 100);
                y = zeros(size(x));
                for l = 1:L
                    y = y + gm.omega(l) * normpdf(x, gm.theta(l), sqrt(gm.phi(l)));
                end
                plot(x,y,'g')
                hold off
                title('GM fit to rhat')
                legend('rhat','GM')
                drawnow;
                pause;
            end
            
        end
        
        % Approximate gradient descent method to minimize empirical SURE
        function lambda = minSureGrad(obj, lambda0, rhat, c, lam_max, lam_min, t, tune_it, debug)
            
            % perform optimization via approx grad
            % descent as described in "Parameterless
            % Optimal Approximate Message Passing" by
            % A. Mousavi, A. Maleki, and R. Baraniuk.
            
            rhat2 = rhat.^2;
            N = numel(rhat);
            
            % options
            %dt = 1;  % for empirical gradient calc (recommended .05-.5)
            maxit = 50; % max number of iterations
            minit = 10;  % min number of iterations
            
            % history over the course of a single
            % GAMP iteration
            lambda_hist = nan(maxit,1);
            step_hist = lambda_hist;
            cost_hist = lambda_hist;
            grad_hist = lambda_hist;
            grad_old = 0;
            lambda = lambda0;
            
            rr = sort(abs(rhat)/c); % sorted points of major change in SURE cost
            tol = 1e-4; % convergence tolerance
            step = obj.sureParams.step(t);
            
            for it=1:maxit
                % compute empirical gradient
                cost = obj.SUREcost(lambda,rhat,rhat2,c);
                cost_hist(it) = cost;
                
                [~,indx]=min(abs( lambda - rr ));
                if indx==1 % detect special case
                    points = 10;
                    cost_left = obj.SUREcost(rr(1:points)+eps,rhat,rhat2,c);
                    [~,p_opt] = min(cost_left);
                    lambda = rr(p_opt);
                    break;
                end
                dt = rr(min(N,indx + 5)) - rr(max(1,indx-5));
                
                grad = (obj.SUREcost(rr(min(N,indx+5)),rhat,rhat2,c) ...
                    - obj.SUREcost(rr(max(1,indx-5)),rhat,rhat2,c))/dt;
                
                grad_hist(it) = grad;
                step = step + 0.1*step*sign(grad)*sign(grad_old);
                if lambda - step*grad < 0
                    step = 0.5*lambda/grad;
                end
                step_hist(it) = step;
                grad_old = grad;
                
                % gradient projection (lambda can't be negative or larger than lambda max)
                lambda_old = lambda;
                lambda = max(lam_min,min(lambda-step*grad, lam_max));
                lambda_hist(it) = lambda;
                
                % check for convergence
                if (it>minit) && (abs(lambda - lambda_old)/lambda_old < tol)
                    break
                end
                
            end
            
            obj.sureParams.step(t) = step;
            
            if debug
                lgrid = logspace(log10(lam_min),log10(lam_max),1e3);
                cost = obj.SUREcost(lgrid,rhat,rhat2,c);
                figure(100);clf;
                semilogx(lgrid,cost,'g')
                hold on
                plot(lambda0,obj.SUREcost(lambda0,rhat,rhat2,c),'go')
                plot(lambda_hist,obj.SUREcost(lambda_hist,rhat,rhat2,c),'bo')
                plot(lambda,obj.SUREcost(lambda,rhat,rhat2,c),'rs')
                hold off
                xlabel('lambda')
                ylabel('sure val')
                title(sprintf('gamp it = %d',tune_it))
                legend('objective','lam0','lam final')
                
                figure(101);clf;
                subplot(411)
                semilogy(lambda_hist,'.-')
                xlabel('iter')
                ylabel('lambda')
                subplot(412)
                semilogy(step_hist,'.-')
                xlabel('iter')
                ylabel('step')
                subplot(413)
                plot(grad_hist,'.-')
                xlabel('iter')
                ylabel('grad')
                subplot(414)
                plot(cost_hist,'.-')
                xlabel('iter')
                ylabel('cost')
                drawnow;
                pause;
            end
            
        end
        
        
        function scost = SUREcost(obj,lambda,rhat,rhat2,c)
            % compute SURE cost using empirical average
            n = numel(rhat);
            scost = nan(size(lambda));
            for ll = 1:numel(lambda)
                scost(ll) = sum(obj.g2(lambda(ll),rhat(:),rhat2(:),c) + 2*c*obj.gp(lambda(ll),rhat(:),c))/n;
            end
        end
        
        function val = gp(~,t,r,c)
            % g-prime
            val = zeros(size(r));
            val(abs(r)<t*c) = -1;
        end
        
        function r2 = g2(~,t,r,r2,c)
            %g-squared
            r2(abs(r)>=t*c) = (t.^2).*(c.^2);
        end
        
        function gm = emgm(obj,rhat,c,L,t)
            % use EM to fit GM distribution to rhat
            
            if L~=3
                error('L must equal 3')
            end
            
            % use EM to fit GM to rhat
            rhat = rhat(:);
            N = numel(rhat);
            maxit = 8;
            tol = 1e-3;
            
            % determine whether to warmstart
            warmstart = 1;
            if isempty(obj.sureParams.GM{t})
                warmstart = 0;
            else
               if all(abs(obj.sureParams.GM{t}.theta) < .1) %% any(obj.sureParams.GM{t}.omega < 1e-4) || 
                  warmstart = 0; 
               end
            end
            
            % random mean initialization/warmstarting
            if ~warmstart  % isempty(obj.sureParams.GM{t}) 
                omega = ones(L,1)/L;
                phi = var(rhat(:))*omega;
                theta = [-0.3333 0 0.3333];
            else
                omega = obj.sureParams.GM{t}.omega;
                theta = obj.sureParams.GM{t}.theta;
                phi = obj.sureParams.GM{t}.phi;
            end
            
            p = nan(N,L);
            twopi = 2*pi;
            g = @(x, m, s) 1/sqrt(twopi * s) * exp(-1/2/s * (x - m).^2);
            omega_hist = nan(L,maxit+1);
            theta_hist = nan(L,maxit+1);
            phi_hist = nan(L,maxit+1);
            
            omega_hist(:,1) = omega';
            theta_hist(:,1) = theta';
            phi_hist(:,1) = phi;
            
            for it = 1:maxit
                
                omega_old = omega;
                theta_old = theta;
                phi_old = phi;
                
                % E-step (slow...)
                for l = 1:L
                    p(:,l) = omega(l)*g(rhat, theta(l), phi(l));
                end
                p(isnan(p)) = 0;
                
                % sum to one
                p = bsxfun(@times, p, 1./sum(p,2));
                
                % M-step
                theta = sum(bsxfun(@times, p, rhat))./sum(p);
                theta(isnan(theta)) = 0;
                
                for l = 1:L
                    phi(l) = sum(p(:,l).*(rhat - theta(l)).^2)/sum(p(:,l));
                end
                phi(isnan(phi)) = 1;
                if obj.sureParams.gm_minvar
                    phi = max(phi, c);
                end
                
                omega = sum(p)/N;
                omega = omega/sum(omega);
                
                omega_hist(:,it+1) = omega';
                theta_hist(:,it+1) = theta';
                phi_hist(:,it+1) = phi;
                
                % check for convergence
                if norm(omega(:)-omega_old(:))/norm(omega_old+eps) < tol && norm(theta(:)-theta_old(:))/norm(theta_old+eps) < tol && norm(phi(:)-phi_old(:))/norm(phi_old+eps) < tol
                    break
                end
            end
            
            gm.omega = omega(:);
            gm.theta = theta(:);
            gm.phi = phi(:);
            
            obj.sureParams.GM{t} = gm;
            
        end
        
        function gm = embgm(obj, rhat, rvar, c, L, t)
            % use EM to fit Bern-GM to xhat (which corresponds to GM on
            % rhat)
            
            rhat = rhat(:);
            rvar = rvar(:);
            N = numel(rhat);
            
            % random mean initialization/warmstarting
            if isempty(obj.sureParams.GM{t})
                lambda = .5; %#ok<*PROP>
                custom_scale = 1;
                load('inits.mat');
                omega = init(L-1).active_weights;
                theta = init(L-1).active_mean;
                phi = init(L-1).active_var;
                if ~isempty(obj.sureParams.initVar)
                    theta = theta*sqrt(12*obj.sureParams.initVar)*custom_scale;
                    phi = phi*12*obj.sureParams.initVar*custom_scale; 
                end
                if ~isempty(obj.sureParams.initSpar)
                   lambda = obj.sureParams.initSpar;
                end
            else
                omega = obj.sureParams.GM{t}.omega;
                theta = obj.sureParams.GM{t}.theta;
                phi = obj.sureParams.GM{t}.phi;
                
                % "deconvolve" with N(0,rvar)
                lambda = 1 - omega(end);
                omega = omega(1:end-1)/lambda;
                theta = theta(1:end-1);
                phi = phi(1:end-1) - phi(end);
                
            end
           
            % expand
            one = ones(N,1,L-1);
            omega = bsxfun(@times, reshape(omega, 1,1,L-1), one);
            theta = bsxfun(@times, reshape(theta, 1,1,L-1), one);
            phi = bsxfun(@times, reshape(phi, 1,1,L-1), one);
            
            D_l = zeros(N,1,L-1); a_nl = zeros(N,1,L-1);
            gamma = zeros(N,1,L-1); nu = zeros(N,1,L-1);
            
            abs_rhat2_over_rvar = abs(rhat).^2./rvar;
            %Evaluate posterior likelihoods
            for i = 1:L-1
                post_var_scale = rvar+phi(:,:,i)+eps;
                rvar_over_post_var_scale = rvar./post_var_scale;
                D_l(:,:,i) = lambda*omega(:,:,i)./sqrt(post_var_scale)...
                    .*exp(-abs(theta(:,:,i)-rhat).^2./(2*post_var_scale));
                gamma(:,:,i) = (rhat.*phi(:,:,i)+rvar.*theta(:,:,i))./post_var_scale; 
                nu(:,:,i) = rvar_over_post_var_scale.*phi(:,:,i);
                a_nl(:,:,i) = sqrt(rvar_over_post_var_scale).*omega(:,:,i)...
                    .*exp((abs(rhat-theta(:,:,i)).^2./abs(post_var_scale)-abs_rhat2_over_rvar)./(-2));  
            end;
            
            %Find posterior that the component x(n,t) is active
            a_n = lambda./(1-lambda).*sum(a_nl,3);
            a_n = 1./(1+a_n.^(-1));
            a_n(isnan(a_n)) = 0.001;
            
            lambda = sum(a_n)/N*ones(N,1);
            
            %Find the Likelihood that component n,t belongs to class l and is active
            E_l = bsxfun(@times, D_l, 1./(sum(D_l,3)+(1-lambda)./sqrt(rvar).*exp(-abs_rhat2_over_rvar/2))); 
            
            %Ensure real valued probability
            E_l(isnan(E_l)) = 0.999;
            
            %Update parameters based on EM equations
            N_l = sum(E_l);
            theta = resize(sum(E_l.*gamma)./N_l,N,1,L-1);
            phi = resize(sum(E_l.*(nu+abs(gamma-theta).^2))./N_l,N,1,L-1);
            omega = N_l/N;
            omega = omega./repmat(sum(omega, 3), [1, 1, L-1]);
            omega = resize(omega,N,1,L-1);
            
            % convolve with N(0,rvar)
            lambda = squeeze(lambda(1,1,:));
            weights = squeeze(omega(1,1,:));
            means = squeeze(theta(1,1,:));
            variances = squeeze(phi(1,1,:));
            
            gm.omega = [lambda * weights; 1-lambda];
            gm.theta = [means;0];
            gm.phi = [variances + c;c];
            
            obj.sureParams.GM{t} = gm;
            
        end
        
        function scost = gmSUREcost(~,lambda,c,gm)
            % compute SURE value using statisical expectation instead of
            % empirical average
            
            omega = gm.omega(:);
            theta = gm.theta(:);
            phi = gm.phi(:);
            
            L = length(omega);
            
            scost = nan(size(lambda));
            c1 = scost;
            c2 = scost;
            c3 = scost;
            
            mu = sum(omega.*theta);
            va = sum(omega.*((theta - mu).^2 + phi)); %#ok<NASGU>
            
            prllc = @(tau, omega, theta, phi) sum(omega.*normcdf(tau, theta, sqrt(phi)));
            prglc = @(tau, omega, theta, phi) 1 - prllc(tau, omega, theta, phi);
            
            for ll = 1:numel(lambda)
                
                tau = lambda(ll)*c;
                val1 = max(0,prglc(tau, omega, theta, phi));
                val2 = max(0,prllc(-tau,omega, theta, phi));
                
                Er2 = 0;
                for l = 1:L
                    b = (tau - theta(l))/sqrt(phi(l));
                    a = (-tau - theta(l))/sqrt(phi(l));
                    phia = normpdf(a);
                    phib = normpdf(b);
                    Phiab = normcdf([a,b]);
                    Phia = Phiab(1);Phib = Phiab(2);
                    %                     Phib = normcdf(b);
                    Z = Phib - Phia;
                    va = phi(l) * ( 1 + (a*phia - b*phib)/Z - ((phia - phib)/Z)^2);
                    mu = theta(l) + (phia - phib)/Z*sqrt(phi(l));
                    Er2 = Er2 + omega(l) * Z * (va + mu^2);
                end
                
                c1(ll) = tau^2 * val1;
                c2(ll) = Er2 - 2*c*(1 - val1 - val2);
                c3(ll) = tau^2 * val2;
                
                scost(ll) =  c1(ll) + c2(ll) + c3(ll);
                
            end
            
        end
        
        function sgrad = gmSUREgrad(~,lambda,c,gm)
            % compute gradient of GM-SURE cost
            
            sgrad = nan(size(lambda));
                        
            % evaluate Gradient
            prllc = @(tau, omega, theta, phi) sum(omega.*normcdf(tau, theta, sqrt(phi)));
            prglc = @(tau, omega, theta, phi) 1 - prllc(tau, omega, theta, phi);
            pprllc = @(tau, omega, theta, phi) sum(omega.*normpdf(tau, theta, sqrt(phi)));
            
            grad1 = @(gm, lam, c) ...
                2 * lam * c^2 .* (prglc(lam*c, gm.omega, gm.theta, gm.phi) + prllc(-lam*c, gm.omega, gm.theta, gm.phi));
            
            grad2 = @(gm, lam, c) ...
                - 2 * c^2 * (pprllc(lam*c, gm.omega, gm.theta, gm.phi) + pprllc(-lam*c, gm.omega, gm.theta, gm.phi));
            
            for ll = 1:numel(lambda)
                sgrad(ll) = grad1(gm, lambda(ll), c) + grad2(gm, lambda(ll), c);
            end
            
        end
        
        
    end
    
end

