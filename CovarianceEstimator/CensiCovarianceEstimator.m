classdef CensiCovarianceEstimator < CovarianceEstimator
    properties
        CovZ     = eye(6);
    end

    methods
        function Sigma = compute(obj, pc1, pc2, T_est)
            %–– correspondence step (unchanged)
            pts1 = pc1.Location;
            pts2 = pctransform(pc2, affine3d(T_est')).Location;
            idx  = knnsearch(pts1, pts2);
            P    = pts1(idx,:);
            Q    = pts2;
            n    = numel(idx);

            % Precompute J_i for each i
            J_all = zeros(6*n,6);
            for i=1:n
              Rp = T_est(1:3,1:3) * P(i,:)';         % R*P_i
              Ji = [ -SE3Utils.skew(Rp), eye(3) ];           % 3×6
              J_all(3*(i-1)+1:3*i, :) = Ji;         % stack
            end
            
            % Analytic Gauss–Newton Hessian
            H_analytic = 2 * (J_all' * J_all);
            
            % Build Cov_z = blockdiag(CovZ,…)
            CovZfull = kron(eye(n), obj.CovZ);
            
            % Analytic cross‐term Jxz = 2 * J_all' * [I_{3n}  0; 0  -I_{3n}]
            % (or more simply use the fact ∂r/∂Q = -I, ∂r/∂P = +I)
            % Splitting CovZ into [CovP, CovPQ; CovQP, CovQ], you get:
            Jxz_analytic = 2 * J_all' * blkdiag( eye(3*n), -eye(3*n) );
            
            % Then finally
            Sigma = H_analytic \ (Jxz_analytic * CovZfull * Jxz_analytic') / H_analytic;
        end
    end
end
