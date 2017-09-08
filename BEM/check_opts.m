% Set the default options andrRead the user input options
%
function [optGAMP, optPE] = check_opts(optGAMP_user, optPE_user)

%Set default GAMP options
optGAMP = GampOpt('nit',1000,...
                    'removeMean',false,...
                    'adaptStep',true,...
                    'adaptStepBethe',true,...
                    'stepWindow',0,...
                    'bbStep',0,...
                    'uniformVariance',0,...
                    'verbose',0,...
                    'tol',1e-6,...
                    'step',0.1,...
                    'stepMin',0,...
                    'stepMax',0.9,...
                    'stepIncr',1.1,...
                    'stepDecr',0.9,...
                    'pvarMin',1e-12,...
                    'xvarMin',1e-12,...
                    'maxBadSteps',inf,...
                    'maxStepDecr',0.9,...
                    'stepTol',1e-12,...
                    'pvarStep',true,...
                    'varNorm',false, ...
                    'valIn0',-Inf);
                    
%Override GAMP defaults if specified by user
if ~isempty(optGAMP_user)
    names = fieldnames(optGAMP_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optGAMP), names{i}));
            optGAMP.(names{i}) = optGAMP_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized GAMP option'])
        end       
    end
end
%Force these GAMP options regardless of what user or default says!
optGAMP.legacyOut = false;
optGAMP.warnOut = false;


%Set default PE options
optPE = PEOpt();

%Change any PE options specified by user
if ~isempty(optPE_user)

    %Check to see if user set any other optional PE parameters
    if ~isfield(optPE_user,'L') % ...if user has no preference about L
        optPE.L = 3;
    end;
    
    if isfield(optPE_user,'L'),
        optPE.L = optPE_user.L;
    end;
    if isfield(optPE_user, 'active_weights'),
        optPE.active_weights = optPE_user.active_weights;
    end;
    if isfield(optPE_user,'noise_var'), 
      optPE.noise_var = optPE_user.noise_var; 
    end;
    if isfield(optPE_user,'lambda'), 
      optPE.lambda = optPE_user.lambda; 
    end;
    if isfield(optPE_user, 'beta')
        optPE.beta = optPE_user.beta;
    end
    if isfield(optPE_user,'cmplx_in')
        optPE.cmplx_in = optPE_user.cmplx_in;
    end
    if isfield(optPE_user,'cmplx_out')
        optPE.cmplx_out = optPE_user.cmplx_out;
    end
    if isfield(optPE_user, 'maxPEiter')
        optPE.maxPEiter = optPE_user.maxPEiter;
    end
   
    %Change the main PE options if specified by user
    names = fieldnames(optPE_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optPE), names{i}));
            optPE.(names{i}) = optPE_user.(names{i});
		else
	    	warning(['''',names{i},''' is an unrecognized PE option'])
        end       
    end
end

return
