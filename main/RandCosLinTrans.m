classdef RandCosLinTrans < LinTrans
    % RandCosLinTrans:  Construct the sensing matrix for image recovery
    % Shuai Huang 
    % 10/26/2016
    
    
    properties
        D;      % The random gaussian measurement matrix
        Dsq;    % Dsq = (D.^2)
        imSize; % image size
        Psi;    % The wavelet transform (matrix)
        Psit;   % The inverse wavelet transform (matrix)
        Psi2;   % The wavelet trasform (matrix) ^2
        Psit2;  % The inverse wavelet transform (matrix) ^2
        matSize;    % The sensing matrix size ...
        Avar;   % Avar...
    end
    
    methods
        
        % Constructor
        function obj = RandCosLinTrans( D, imSize, Psi, Psit, Psi2, Psit2, matSize, Avar)
            obj = obj@LinTrans;
            obj.D = D;
            obj.Dsq = (abs(D).^2);
            obj.imSize = imSize;
            obj.Psi = Psi;
            obj.Psit = Psit;
            obj.Psi2 = Psi2;
            obj.Psit2 = Psit2;
            obj.matSize = matSize;
            
            %Assign matrix entry-wise variances if provided
            if nargin < 8
                obj.Avar = 0;
            else
                obj.Avar = Avar;
            end
        end
        
        % size method ( deals with optional dimension argin  ; nargout={0,1,2} )
        function [m,n] = size(obj,dim)
            if nargin>1 % a specific dimension was requested
                if dim>2
                    m=1;
                else
                    m=obj.matSize(dim);
                end
            elseif nargout<2  % all dims in one output vector
                m=obj.matSize;
            else % individual outputs for the dimensions
                m = obj.matSize(1);
                n = obj.matSize(2);
            end
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            x_mat = obj.Psit(x);
            y = obj.D*reshape(x_mat, numel(x_mat), 1);
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            y_mat = reshape(obj.D'*x, obj.imSize);
            y = obj.Psi(y_mat);
        end
        % Matrix multiply with square
        function y = multSq(obj,x)
            x_mat = obj.Psit2(x);
            y = obj.Dsq*reshape(x_mat, numel(x_mat), 1);
        end
        % Matrix multiply transpose
        function y = multSqTr(obj,x)
            y_mat = reshape(obj.Dsq'*x, obj.imSize);
            y = obj.Psi2(y_mat);
        end
        
    end
end
