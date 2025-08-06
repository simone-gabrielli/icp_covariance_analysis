% Works well with rotations, translations do not vary
classdef WeightedHessian < CovarianceEstimator
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
        
            % extract R and t
            R = T_est(1:3,1:3);
            t = T_est(1:3,4);
        
            % build analytic J_all (3n×6) and Hessian H
            J_all = zeros(3*n,6);
            for i=1:n
                Rp   = R * P(i,:)';
                Ji   = [ -SE3Utils.skew(Rp), eye(3) ];  % 3×6
                J_all(3*(i-1)+1:3*i, :) = Ji;
            end
            H = 2 * (J_all' * J_all);  % 6×6 analytic Gauss–Newton Hessian
        
            % residuals
            diffs = (R*P' + t)' - Q;       % n×3
            dists = vecnorm(diffs,2,2);    % n×1
        
            % build an error‐ and orientation‐dependent CovZfull
            CovZfull = zeros(6*n);
            for i=1:n
                % rotate your base CovZ into world
                % assume obj.CovZ = [ Cp    0
                %                     0     Cq ]
                Cp = obj.CovZ(1:3,1:3);
                Cq = obj.CovZ(4:6,4:6);
                Cp_w = R * Cp * R';   % 3×3
                Cq_w = R * Cq * R';   % 3×3
        
                % scale by squared residual
                W = dists(i)^2;
        
                block = W * [ Cp_w,    zeros(3);
                              zeros(3), Cq_w ];  % 6×6
        
                ix = 6*(i-1)+(1:6);
                CovZfull(ix,ix) = block;
            end
        
            % assemble analytic cross‐Jacobian (6×6n)
            Jxz = zeros(6,6*n);
            for i=1:n
                Ji    = J_all(3*(i-1)+1:3*i, :);    % 3×6
                block = 2 * Ji' * [ eye(3), -eye(3) ];  % 6×6
                Jxz(:,6*(i-1)+(1:6)) = block;
            end
        
            % final covariance
            Sigma = H \ (Jxz * CovZfull * Jxz') / H;
        end


    end
end
