% The Generalized approximate message passing with built-in parameter restimation (PE-GAMP)
%
% The simplified PE-GAMP where the parameters at different variable nodes are the same
%
% Max-product message passing with Laplace prior
% For Laplace prior, max-product MP doesn't produce any meaningful estimation of the parameters, for details please refer to the paper.
%
% Shuai Huang, The Johns Hopkins University
% E-mail: shuang40@jhu.edu
% Date: 09/03/2017
%

function [Xhat, PEfin, estHist] = MAX_LP_PE_GAMP(Y, A, optPE, optGAMP)

% If A is an explicit matrix, replace by an operator
if isa(A, 'double')
    A = MatrixLinTrans(A);
end

%Find problem dimensions
[M,N] = A.size();
T = size(Y, 2);
if size(Y, 1) ~= M
    error('Dimension mismatch betweeen Y and A')
end

%Merge user-specified GAMP and PE options with defaults
if nargin <= 2
    optPE = [];
end
if nargin <= 3
    optGAMP = [];
end
[optGAMP, optPE] = check_opts(optGAMP, optPE);

%Initialize PE parameters
[lambda, optPE] = set_initsLP(optPE, Y, A, M, N, T);
lambda_all = lambda(1,1);

noise_var = optPE.noise_var;
noise_var_all = noise_var(1,1);


%Handle history saving/reporting
histFlag = false;
if nargout >=3
    histFlag = true;
    estHist = [];
end;

%Initialize loop
firstStep = optGAMP.step;
t = 0;
stop = 0;


%Initialize XhatPrev
XhatPrev = inf(N,T);

while stop == 0
    %Increment time exit loop if exceeds maximum time
    t = t + 1;
    
    if t >= optPE.maxPEiter
        stop = 1;
    end
    
    %Input channel for real or complex signal distributions
    if ~optPE.cmplx_in
        inputEst = SoftThreshEstimIn(lambda, true); 
    else
        % TBD
    end

    %Output channel for real or complex noise distributions
    if ~optPE.cmplx_out
        outputEst = AwgnEstimOut(Y, noise_var, true);
    else
        % TBD
    end

    %Perform GAMP
    if ~histFlag
        estFin = gampEst(inputEst, outputEst, A, optGAMP);
    else
        [estFin,~,estHistNew] = gampEst(inputEst, outputEst, A, optGAMP);
        estHist = appendEstHist(estHist,estHistNew);
    end

    Xhat = estFin.xhat;
    Xvar = estFin.xvar;
    Rhat = estFin.rhat;
    %If rhats are returned as NaN, then gamp failed to return a better
    %estimate,  PE has nothing to go on, so break.

    if any(isnan(Rhat(:))); break; end;
    Rvar = estFin.rvar;
    Phat = estFin.phat;
    Pvar = estFin.pvar;

    %Update parameters for either real or complex signal distributions
    if ~optPE.cmplx_in
        % Doesn't work this way
    else
        % TBD
    end

    %Update noise variance.
    if ~optPE.cmplx_out
        % Doesn't work this way
    else
        % TBD
    end
    

    %Calculate the change in signal estimates
    norm_change = norm(Xhat-XhatPrev,'fro')/norm(Xhat,'fro');

    %Check for estimate tolerance threshold
    if norm_change < optPE.PEtol
        stop = 1;
    end

    XhatPrev = Xhat;
    % Warm-start reinitialization of GAMP
    % This is very important for the stabality of the algorithm
    optGAMP = optGAMP.warmStart(estFin);

end;

%Output final solution 
Xhat = estFin.xhat;

%Output final parameter estimates
PEfin.Xvar = estFin.xvar;
PEfin.Zhat = estFin.zhat;
PEfin.Zvar = estFin.zvar;
PEfin.Rhat = estFin.rhat;
PEfin.Rvar = estFin.rvar;
PEfin.Phat = estFin.phat;
PEfin.Pvar = estFin.pvar;

% Laplace prior parameters
PEfin.lambda = lambda;
PEfin.noise_var = noise_var;

PEfin.lambda_all = lambda_all;
PEfin.noise_var_all = noise_var_all;

return;
