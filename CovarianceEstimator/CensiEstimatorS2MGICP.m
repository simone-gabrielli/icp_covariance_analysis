classdef CensiEstimatorS2MGICP < CovarianceEstimator
    properties
        CovZ      = eye(3) * 0.1;
        H_sym     % symbolic Hessian d^2J/dX^2  (optional to keep around)
        Jz_sym    % symbolic ∂g/∂Z            (optional to keep around)
        subs_vars % list of syms for fast subs() (optional)
        fH        % matlabFunction handle for H_sym
        fJz       % matlabFunction handle for Jz_sym

        K = 10; % number of neighbors for covariance estimation
    end

    methods
        function obj = CensiEstimatorS2MGICP()
            % Symbolic variables are not used in this code,
            % Ctor was kept for compatibility with the original code structure.
            % and to show the mathematical formulation.

            import sympy.*;
            syms x y z a b c pix piy piz qix qiy qiz real
            % GICP covariance placeholders (symbolic 3x3 SPD matrices)
            syms Cp [3 3] real
            syms Cq [3 3] real

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
            % GICP: weight by combined covariance W = (R*Cp*R' + Cq)^{-1}
            W    = inv(Rmat*Cp*Rmat.' + Cq);
            J    = err.' * W * err;
            % g    = jacobian(J, [x; y; z; a; b; c]).';
            % H    = hessian(J, [x; y; z; a; b; c]);
            % Jz   = jacobian(g, [pix; piy; piz]);

            % % store the raw symbols (optional)
            % obj.H_sym     = H;
            % obj.Jz_sym    = Jz;
            % obj.subs_vars = {pix, piy, piz, qix, qiy, qiz, x, y, z, a, b, c, Cp, Cq};

            % % compile into fast function-handles
            % obj.fH  = matlabFunction(H,  'Vars', obj.subs_vars);
            % obj.fJz = matlabFunction(Jz, 'Vars', obj.subs_vars);
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

            % Compute points covariance with knn 
            Cp_all = zeros(3,3,n);
            Cq_all = zeros(3,3,n);

            % For each point in P and Q, estimate local covariance using kNN
            parfor i = 1:n
                % For P
                [idxP, ~] = knnsearch(P, P(i,:), 'K', obj.K+1); % include self
                neighborsP = P(idxP(2:end), :); % exclude self
                Cp = cov(neighborsP);
                if rank(Cp) < 3
                    Cp = Cp + 1e-6*eye(3); % regularize if degenerate
                end
                Cp_all(:,:,i) = Cp;

                % For Q
                [idxQ, ~] = knnsearch(Q, Q(i,:), 'K', obj.K+1);
                neighborsQ = Q(idxQ(2:end), :);
                Cq = cov(neighborsQ);
                if rank(Cq) < 3
                    Cq = Cq + 1e-6*eye(3);
                end
                Cq_all(:,:,i) = Cq;
            end

            %% NOT SURE OF THIS MATHEMATICAL FORMULATION, BETTER TRY THE FORMULAS OBTAINED
            %% FROM EXACT COMPUTATION IN CONSTRUCTOR

            % preallocate (numerical accumulation)
            d2J_dX2  = zeros(6,6);
            d2J_dZdX = zeros(6,3*n);

            % helper for skew matrix
            skew = @(v) [    0   -v(3)  v(2);
                             v(3)   0  -v(1);
                            -v(2)  v(1)   0 ];

            % use current estimate's rotation to combine covariances (no differentiation)
            R_est = T_est(1:3,1:3);

            for i = 1:n
                p = P(i,:).';
                q = Q(i,:).';
                r = p - q; % residual at identity linearization

                % Jacobian wrt state X=[x;y;z;a;b;c]
                Jx = [eye(3), -skew(p)]; % 3x6

                % Combined covariance and its Cholesky factor
                S = R_est*Cp_all(:,:,i)*R_est.' + Cq_all(:,:,i);
                % ensure SPD (in case of numerical issues)
                [L,flag] = chol(S,'lower');
                if flag ~= 0
                    S = S + 1e-9*eye(3);
                    L = chol(S,'lower');
                end

                % Compute W*Jx and W*r via triangular solves (avoid explicit inverse)
                WJx = L.' \ (L \ Jx);
                Wr  = L.' \ (L \ r);

                % Accumulate Gauss-Newton Hessian: H += 2 * Jx' * W * Jx
                d2J_dX2 = d2J_dX2 + 2 * (Jx.' * WJx);

                % Jz for this correspondence (6x3):
                % 2 * (Jx' * W) + 2 * [0; -skew(W*r)]
                Jz_i = 2 * (WJx.') + [zeros(3,3); -2 * skew(Wr)];

                % place block
                d2J_dZdX(:, 3*(i-1)+1 : 3*i) = Jz_i;
            end

            % form measurement-noise cov and final ICP covariance
            bigZ  = kron(eye(n), obj.CovZ);
            % avoid explicit inverse
            invHX = d2J_dX2 \ eye(6);
            Sigma = invHX * d2J_dZdX * bigZ * d2J_dZdX.' * invHX;

            % reorder both rows & cols so params go [a,b,c,x,y,z], as per convention
            perm  = [4 5 6 1 2 3];
            Sigma = Sigma(perm,perm);
        end
    end
end