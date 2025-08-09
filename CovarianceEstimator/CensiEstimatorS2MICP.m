classdef CensiEstimatorS2MICP < CovarianceEstimator
    properties
        CovZ      = eye(3) * 0.1;
        H_sym     % symbolic Hessian d^2J/dX^2  (optional to keep around)
        Jz_sym    % symbolic ∂g/∂Z            (optional to keep around)
        subs_vars % list of syms for fast subs() (optional)
        fH        % matlabFunction handle for H_sym
        fJz       % matlabFunction handle for Jz_sym
    end

    methods
        function obj = CensiEstimatorS2MICP()
            import sympy.*;
            syms x y z a b c pix piy piz qix qiy qiz real

            % build the J, g, H, Jz
            % Method 1) RPY
            Rz = [ cos(a) -sin(a) 0;
                   sin(a)  cos(a) 0;
                     0        0   1];
            Ry = [ cos(b) 0 sin(b);
                     0    1   0;
                  -sin(b) 0 cos(b)];
            Rx = [ 1   0       0;
                   0 cos(c) -sin(c);
                   0 sin(c)  cos(c)];
            Rmat = Rz*Ry*Rx;
            % Method 2) use Taylor‐second‐order parametrization for consistency
            % S    = [0 -c b; c 0 -a; -b a 0];
            % Rmat = eye(3) + S + 0.5*(S*S);
            p    = [pix; piy; piz];
            q    = [qix; qiy; qiz];
            Tp   = Rmat*p + [x; y; z];
            err  = Tp - q;
            J    = err.'*err;
            g    = jacobian(J, [x; y; z; a; b; c]).';
            H    = hessian(J, [x; y; z; a; b; c]);
            Jz   = jacobian(g, [pix; piy; piz]);

            % store the raw symbols (optional)
            obj.H_sym     = H;
            obj.Jz_sym    = Jz;
            obj.subs_vars = {pix, piy, piz, qix, qiy, qiz, x, y, z, a, b, c};

            % compile into fast function-handles
            obj.fH  = matlabFunction(H,  'Vars', obj.subs_vars);
            obj.fJz = matlabFunction(Jz, 'Vars', obj.subs_vars);
        end

        function Sigma = compute(obj, pc1, pc2, T_est)
            %–– correspondence step ––
            pts1 = pc1.Location;
            pts2 = pctransform(pc2, affine3d(T_est')).Location;
            idx  = knnsearch(pts2, pts1);
            P    = pts1;
            Q    = pts2(idx,:);
            n    = numel(idx);
            X    = zeros(6,1);

            % P = [-0.5 -0.5, 0;
            %      -0.5, 0.5, 0;
            %      0.5, -0.5, 0;
            %      0.5, 0.5, 0];
            % Q = [-0.5 -0.5, 1;
            %      -0.5, 0.5, 1;
            %      0.5, -0.5, 1;
            %      0.5, 0.5, 1];
            % n = 4;

            % The code below works propely with 2 offseted planes with given correspondences,
            % but with the matching above not so well.

            % Visualize correspondences
            % figure;
            % pcshow(pc1);
            % hold on;
            % pcshow(pc2.Location, 'r');
            % for i = 1:n
            %     plot3([P(i,1) Q(i,1)], [P(i,2) Q(i,2)], [P(i,3) Q(i,3)], 'g-');
            % end
            % legend('pc1','pc2','Correspondences');
            % hold off;

            % preallocate
            d2J_dX2  = zeros(6,6);
            d2J_dZdX = zeros(6,3*n);

            for i = 1:n
                % pack arguments in the same order as subs_vars
                args = { P(i,1), P(i,2), P(i,3), ...
                         Q(i,1), Q(i,2), Q(i,3), ...
                         X(1),   X(2),   X(3),   ...
                         X(4),   X(5),   X(6) };
                Hi  = obj.fH(args{:});
                Jzi = obj.fJz(args{:});
                d2J_dX2 = d2J_dX2 + Hi;
                d2J_dZdX(:,3*(i-1)+1:3*i) = Jzi;
            end

            % form measurement-noise cov and final ICP covariance
            bigZ  = kron(eye(n), obj.CovZ);
            invHX = inv(d2J_dX2);
            Sigma = invHX * d2J_dZdX * bigZ * d2J_dZdX.' * invHX;

            % reorder both rows & cols so params go [a,b,c,x,y,z], as per convention
            perm  = [4 5 6 1 2 3];
            Sigma = Sigma(perm,perm);
        end
    end
end