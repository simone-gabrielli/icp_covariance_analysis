%%========================%%
% MonteCarloValidator.m
%%========================%%
classdef MonteCarloValidator
    properties
        Generator 
        Estimator
        NumSamples (1,1) double = 1000;
    end
    methods
        function obj = MonteCarloValidator(gen, est, varargin)
            % Constructor: requires a generator and an estimator
            obj.Generator = gen;
            obj.Estimator = est;
            % Optional: override number of Monte Carlo samples
            for k=1:2:numel(varargin)
                switch lower(varargin{k})
                    case 'numsamples'
                        obj.NumSamples = varargin{k+1};
                    otherwise
                        error('Unknown parameter %s', varargin{k});
                end
            end
        end

        function results = validate(obj)
            % GENERATE data
            [pc1, pc2, T_true] = obj.Generator.generate();
            % OPTIONALLY align via ICP or direct use of T_true
            T_est = T_true; % or apply alignment algorithm here
            % ESTIMATE covariance
            Sigma_est = obj.Estimator.compute(pc1, pc2, T_est);

            % MONTE CARLO sampling in se(3)
            N = obj.NumSamples;
            residuals = zeros(6, N);
            for i = 1:N
                % sample xi ~ N(0, Sigma_est)
                xi = mvnrnd(zeros(6,1), Sigma_est)';
                dT = SE3Utils.expMapSE3(xi);
                T_pert = dT * T_true;
                % compute error twist: log(inv(T_pert)*T_true)
                T_err   = SE3Utils.invMat(T_pert) * T_true;
                xi_err  = SE3Utils.se3Log(T_err);
                residuals(:,i) = xi_err;
            end

            % empirical covariance and Mahalanobis distances
            results.SigmaEmpirical = cov(residuals');
            results.Mahalanobis   = sum((residuals' / Sigma_est) .* residuals', 2);
        end
    end
end
