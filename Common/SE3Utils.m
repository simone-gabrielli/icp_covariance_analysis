classdef SE3Utils
    % SE3Utils
    % Collection of SE(3) helper functions: exponential/log maps, inversion, skew.
    
    methods(Static)
        function T = expMapSE3(xi)
            % expMapSE3  Exponential map from se(3) vector to SE(3) matrix
            %   xi = [ω; ν], 6×1 twist
            omega   = xi(1:3);
            upsilon = xi(4:6);
            theta   = norm(omega);
            if theta < eps
                R = eye(3); V = eye(3);
            else
                axis = omega / theta;
                R    = axang2rotm([axis', theta]);
                Omega = SE3Utils.skew(omega);
                V     = eye(3) + (1-cos(theta))/(theta^2)*Omega + ...
                        (theta-sin(theta))/(theta^3)*(Omega*Omega);
            end
            t = V * upsilon;
            
            T = eye(4);
            T(1:3,1:3) = R;
            T(1:3,4)   = t;
        end
        
        function xi = se3Log(T)
            % se3Log  Log map from SE(3) to se(3)
            R = T(1:3,1:3);
            t = T(1:3,4);
            % rotm2axang returns a 1×4 vector [ux uy uz theta]
            axang = rotm2axang(R);
            axis  = axang(1:3)';
            theta = axang(4);
            omega = axis * theta;  % 3×1
    
            if theta < eps
                Vinv = eye(3);
            else
                Omega = SE3Utils.skew(omega);
                Vinv  = eye(3) - 0.5*Omega + ...
                    (1/theta^2)*(1 - (theta*sin(theta)/(2*(1-cos(theta)))))*(Omega*Omega);
            end
            upsilon = Vinv * t;
            xi = [omega; upsilon];
        end
        
        function Tinv = invMat(T)
            % invMat  Fast inverse of a homogeneous transform
            R = T(1:3,1:3);
            t = T(1:3,4);
            Tinv = eye(4);
            Tinv(1:3,1:3) = R';
            Tinv(1:3,4)   = -R' * t;
        end
        
        function S = skew(v)
            % skew  Skew-symmetric matrix from a 3-vector
            S = [   0   -v(3)  v(2);
                  v(3)    0   -v(1);
                 -v(2)  v(1)    0   ];
        end
    end
end
