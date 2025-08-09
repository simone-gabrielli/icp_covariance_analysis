% Works well with rotations, translations do not vary
classdef BevingtonRobinsonEstimator < CovarianceEstimator
    properties
        CovZ     = eye(6);
    end

    methods
        function Sigma = compute(obj, pc1, pc2, T_est)
            %–––– correspondence step
            pts1 = pc1.Location;
            pts2 = pctransform(pc2, affine3d(T_est')).Location;
            idx  = knnsearch(pts1, pts2);
            P    = pts1(idx,:);
            Q    = pts2;
            n    = numel(idx);
        
            % extract R,t
            R = T_est(1:3,1:3);
            t = T_est(1:3,4);
        
            % build residual vector (3n×1) and Jacobian (3n×6)
            e_all  = zeros(3*n,1);
            J_all  = zeros(3*n,6);
            for i = 1:n
                Rp = R * P(i,:)';
                ri = P(i,:)'- Q(i,:)';          % 3×1 residual
                e_all(3*(i-1)+1:3*i) = ri;        % stack
        
                Ji = [ -SE3Utils.skew(Rp), eye(3) ];  % 3×6
                J_all(3*(i-1)+1:3*i, :) = Ji;
            end
        
            % Gauss–Newton approximate Hessian
            H = J_all' * J_all;            % 6×6
        
            % gradient
            g = J_all' * e_all;            % 6×1
        
            % empirical variance: sum of squares / dof
            dof    = 3*n - 6;
            %sigma2 = (g' * g) / dof;       % scalar
            sigma2 = (e_all' * e_all) / dof;       % scalar
        
            % covariance = sigma^2 * inv(H)
            Sigma = sigma2 * inv(H);
        end


    end
end
