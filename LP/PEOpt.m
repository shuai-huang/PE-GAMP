% Construct PE options
%

function optPE = PEOpt()

%maximum number of PE iterations.
optPE.maxPEiter = 100;

%maximum tolerance to exit PE loop.
optPE.PEtol = 1e-6;

%Set default number of mixture components
optPE.L = 3;

%Set minium allowed variance of a GM component
optPE.minVar = 1e-12;

return
