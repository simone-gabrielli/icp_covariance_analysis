%%========================%%
% MonteCarloValidator.m
%%========================%%
classdef MonteCarloValidator
    properties
        Generator 
        Estimator
        NumSamples (1,1) double = 1000;
        SigmaSqRot (3,1) double = ones(3,1);
        SigmaSqTrans (3,1) double = ones(3,1);
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
                    case 'sigma_sq_rot'
                        val = varargin{k+1};
                        if isscalar(val)
                            obj.SigmaSqRot = val * ones(3,1);
                        elseif isvector(val) && numel(val) == 3
                            obj.SigmaSqRot = val(:);
                        else
                            error('SigmaSqRot must be a scalar or 3-element vector');
                        end
                    case 'sigma_sq_trans'
                        val = varargin{k+1};
                        if isscalar(val)
                            obj.SigmaSqTrans = val * ones(3,1);
                        elseif isvector(val) && numel(val) == 3
                            obj.SigmaSqTrans = val(:);
                        else
                            error('SigmaSqTrans must be a scalar or 3-element vector');
                        end
                    otherwise
                        error('Unknown parameter %s', varargin{k});
                end
            end
        end

        function results = validate(obj)
            % Generate data
            [pc1, pc2, T_true] = obj.Generator.generate();
            % Ensure transform maps pc2 -> pc1 (estimator warps pc2 towards pc1)
            T_true_21 = inv(T_true);

            N = obj.NumSamples;
            results.Transforms = cell(1, N);
            results.Covariances = cell(1, N);

            for i = 1:N
                % Sample a random perturbation in se(3)
                ri = mvnrnd(zeros(3,1), diag(obj.SigmaSqRot))'; % 0-mean, covariance diag(SigmaSqRot)
                ti = mvnrnd(zeros(3,1), diag(obj.SigmaSqTrans))'; % 0-mean, covariance diag(SigmaSqTrans)
                xi = [ri; ti]; % se(3) perturbation vector
                dT = SE3Utils.expMapSE3(xi);
                % Compose with pc2 -> pc1 ground truth
                T_pert = dT * T_true_21;

                % Estimate covariance for this alignment
                Sigma_est = obj.Estimator.compute(pc1, pc2, T_pert);

                results.Transforms{i} = T_pert;
                results.Covariances{i} = Sigma_est;
            end
        end
    end
end
