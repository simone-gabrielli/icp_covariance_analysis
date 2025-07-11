%%========================%%
% CensiCovarianceEstimator.m
%%========================%%
classdef CensiCovarianceEstimator < CovarianceEstimator
    % Implements the closed-form covariance estimate from Manoj et al. (2015)
    properties
        % Measurement noise covariance per point correspondence (6×6)
        CovZ = eye(6);
        % Small perturbation for numeric differentiation
        Epsilon = 1e-4;
    end
    methods
        function Sigma = compute(obj, pc1, pc2, T_est)
            % 1) correspondences
            pts1 = pc1.Location;
            pts2 = pctransform(pc2, affine3d(T_est')).Location;
            idx  = knnsearch(pts1, pts2);
            P    = pts1(idx,:);
            Q    = pts2;
            n    = size(P,1);
    
            % 2) pack measurements
            z0 = reshape([P, Q]', 6*n, 1);
    
            % 3) nominal twist
            x0 = SE3Utils.se3Log(T_est);
    
            % 4) cost handle
            funJ = @(x,z) obj.Jcost(x,z);
    
            % 5) Hessian H_xx = ∂²J/∂x²
            H   = obj.numericHessian(@(x) funJ(x, z0), x0);
    
            % 6) cross‐Jacobian J_zx = ∂²J/∂z∂x
            Jzx = obj.numericCrossJacobian(funJ, x0, z0);
    
            % 7) full Cov(z)
            CovZfull = kron(eye(n), obj.CovZ);
    
            % 8) apply theorem
            Sigma = H \ (Jzx * CovZfull * Jzx') / H;
        end
    end
    
    methods (Access=private)
        function H = numericHessian(obj, f, x0)
            % Central‐difference Hessian using obj.Epsilon
            m = numel(x0);
            H = zeros(m);
            h = obj.Epsilon;
            for i=1:m
                ei = zeros(m,1); ei(i)=h;
                for j=i:m
                    ej = zeros(m,1); ej(j)=h;
                    fpp = f(x0+ ei+ ej);
                    fpm = f(x0+ ei- ej);
                    fmp = f(x0- ei+ ej);
                    fmm = f(x0- ei- ej);
                    Hij = (fpp - fpm - fmp + fmm)/(4*h^2);
                    H(i,j) = Hij;
                    H(j,i) = Hij;
                end
            end
        end
    
        function g = numericGradient(obj, f, x0)
            % Central‐difference gradient using obj.Epsilon
            m = numel(x0);
            g = zeros(m,1);
            h = obj.Epsilon;
            for i=1:m
                ei = zeros(m,1); ei(i)=h;
                g(i) = (f(x0+ei) - f(x0-ei))/(2*h);
            end
        end
    
        function Jzx = numericCrossJacobian(obj, funJ, x0, z0)
            % ∂²J/∂z∂x via finite differences on z
            m = numel(x0);
            p = numel(z0);
            Jzx = zeros(m, p);
            h = obj.Epsilon;
            for k = 1:p
                dz = zeros(p,1); dz(k)=h;
                grad_p = obj.numericGradient(@(x) funJ(x, z0 + dz), x0);
                grad_m = obj.numericGradient(@(x) funJ(x, z0 - dz), x0);
                Jzx(:,k) = (grad_p - grad_m)/(2*h);
            end
        end
    
        function J = Jcost(obj, x, z)
            % unpack and sum‐of‐squares
            zs = reshape(z, 6, [])';
            P  = zs(:,1:3);
            Q  = zs(:,4:6);
            T  = SE3Utils.expMapSE3(x);
            R  = T(1:3,1:3);  t = T(1:3,4);
            Ptx = (R*P') + t;        % 3×N
            diffs = Ptx' - Q;        % N×3
            J = sum(vecnorm(diffs,2,2).^2);
        end
    end
end