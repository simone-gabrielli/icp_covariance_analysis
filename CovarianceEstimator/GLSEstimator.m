classdef GLSEstimator < CovarianceEstimator
    properties
        CovZ      = eye(3) * 0.1;

        K = 10; % number of neighbors for covariance estimation
    end

    methods
        function C = regCov(obj, C)
            C = (C+C')/2;
            [V,D] = eig(C);
            ev = diag(D);
            ev = max(ev, 1e-3*max(ev));   % floor at 0.1% of max eigenvalue (tune)
            C = V*diag(ev)*V';
        end

        function L = infoChol(obj, Cp, Cq, R)
            M = R*Cp*R.' + Cq;
            M = (M+M.')/2;                            % symmetrize
            [U,p] = chol(M);                          % U'*U = M (upper)
            if p>0, U = chol(M + 1e-9*eye(3)); end    % tiny jitter if needed
            L = U \ eye(3);                           % sqrt-information, L'*L = inv(M)
        end

        function H = buildH(obj, p, Cp, q, Cq, T)
            
            R = T(1:3,1:3);
            eulZYX = rotm2eul(R, 'ZYX'); % Todo check ZYX
            a = eulZYX(1);
            b = eulZYX(2);
            c = eulZYX(3);
            x = T(1,4);
            y = T(2,4);
            z = T(3,4);

            pix = p(1);
            piy = p(2);
            piz = p(3);
            qix = q(1);
            qiy = q(2);
            qiz = q(3);

            % Build GICP information matrix
            L = obj.infoChol(Cp, Cq, R);
            lixx = L(1,1);
            lixy = L(1,2);
            lixz = L(1,3);
            liyy = L(2,2);
            liyz = L(2,3);
            lizz = L(3,3);

            % Build H
            H = zeros(6,6);

            H(1,1) = 2*lixx^2;
            H(1,2) = 2*lixx*lixy;
            H(1,3) = 2*lixx*lixz;
            H(1,4) = 2*lixx*(lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));
            H(1,5) = 2*lixx*(lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));
            H(1,6) = 2*lixx*(lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));

            H(2,1) = 2*lixx*lixy;
            H(2,2) = 2*lixy^2 + 2*liyy^2;
            H(2,3) = 2*lixy*lixz + 2*liyy*liyz;
            H(2,4) = 2*lixy*(lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) + 2*liyy^2*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            H(2,5) = 2*lixy*(lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) + 2*liyy*(liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)));
            H(2,6) = 2*lixy*(lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - 2*liyy*(liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));

            H(3,1) = 2*lixx*lixz;
            H(3,2) = 2*lixy*lixz + 2*liyy*liyz;
            H(3,3) = 2*lixz^2 + 2*liyz^2 + 2*lizz^2;
            H(3,4) = 2*lixz*(lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) + 2*liyy*liyz*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            H(3,5) = 2*lixz*(lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - 2*lizz^2*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*liyz*(liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)));
            H(3,6) = 2*lizz^2*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)) - 2*liyz*(liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + 2*lixz*(lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));

            H(4,1) = lixx*(2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));
            H(4,2) = lixy*(2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) + 2*liyy^2*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            H(4,3) = lixz*(2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) + 2*liyy*liyz*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            H(4,4) = (lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) + 2*liyy^2*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b))^2 - (2*lixx*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + 2*lixy*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));
            H(4,5) = (2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*lixx*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixy*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*liyy*(liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c));
            H(4,6) = (2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*lixx*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixy*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*liyy*(liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));

            H(5,1) = lixx*(2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));
            H(5,2) = lixy*(2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) + liyy*(2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)));
            H(5,3) = lixz*(2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - 2*lizz^2*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + liyz*(2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)));
            H(5,4) = (lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*lixx*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixy*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + liyy*(2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c));
            H(5,5) = (lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*lixy*(pix*cos(b)*sin(a) + piz*cos(c)*sin(a)*sin(b) + piy*sin(a)*sin(b)*sin(c)) + 2*lixz*(piz*cos(b)*cos(c) - pix*sin(b) + piy*cos(b)*sin(c)) + 2*lixx*(pix*cos(a)*cos(b) + piz*cos(a)*cos(c)*sin(b) + piy*cos(a)*sin(b)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*lizz^2*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c))^2 - (liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(2*liyy*(pix*cos(b)*sin(a) + piz*cos(c)*sin(a)*sin(b) + piy*sin(a)*sin(b)*sin(c)) + 2*liyz*(piz*cos(b)*cos(c) - pix*sin(b) + piy*cos(b)*sin(c))) + (liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c))) - 2*lizz^2*(piz*cos(b)*cos(c) - pix*sin(b) + piy*cos(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));
            H(5,6) = (2*lixx*(piy*cos(a)*cos(b)*cos(c) - piz*cos(a)*cos(b)*sin(c)) - 2*lixz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) + 2*lixy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (2*liyz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) - 2*liyy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - 2*lizz^2*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - 2*lizz^2*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));

            H(6,1) = lixx*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));
            H(6,2) = lixy*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - liyy*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));
            H(6,3) = 2*lizz^2*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)) - liyz*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + lixz*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));
            H(6,4) = (lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*lixx*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixy*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - liyy*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            H(6,5) = (2*lixx*(piy*cos(a)*cos(b)*cos(c) - piz*cos(a)*cos(b)*sin(c)) - 2*lixz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) + 2*lixy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (2*liyz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) - 2*liyy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - 2*lizz^2*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - 2*lizz^2*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));
            H(6,6) = (2*lixx*(piy*cos(a)*cos(b)*cos(c) - piz*cos(a)*cos(b)*sin(c)) - 2*lixz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) + 2*lixy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (2*liyz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) - 2*liyy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - 2*lizz^2*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - 2*lizz^2*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));
        end

        function Jz = buildJz(obj, p, Cp, q, Cq, T)

            R = T(1:3,1:3);
            eulZYX = rotm2eul(R, 'ZYX'); % Todo check ZYX
            a = eulZYX(1);
            b = eulZYX(2);
            c = eulZYX(3);
            x = T(1,4);
            y = T(2,4);
            z = T(3,4);

            pix = p(1);
            piy = p(2);
            piz = p(3);
            qix = q(1);
            qiy = q(2);
            qiz = q(3);

            % Build GICP information matrix
            L = obj.infoChol(Cp, Cq, R);
            lixx = L(1,1);
            lixy = L(1,2);
            lixz = L(1,3);
            liyy = L(2,2);
            liyz = L(2,3);
            lizz = L(3,3);

            Jz = zeros(6,3);

            Jz(1,1) = 2*lixx*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a));
            Jz(1,2) = 2*lixx*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c));
            Jz(1,3) = 2*lixx*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c));

            Jz(2,1) = 2*lixy*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a)) - 2*liyy*(liyz*sin(b) - liyy*cos(b)*sin(a));
            Jz(2,2) = 2*lixy*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c)) + 2*liyy*(liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + liyz*cos(b)*sin(c));
            Jz(2,3) = 2*lixy*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c)) - 2*liyy*(liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - liyz*cos(b)*cos(c));

            Jz(3,1) = - 2*sin(b)*lizz^2 - 2*liyz*(liyz*sin(b) - liyy*cos(b)*sin(a)) + 2*lixz*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a));
            Jz(3,2) = 2*cos(b)*sin(c)*lizz^2 + 2*lixz*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c)) + 2*liyz*(liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + liyz*cos(b)*sin(c));
            Jz(3,3) = 2*cos(b)*cos(c)*lizz^2 + 2*lixz*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c)) - 2*liyz*(liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - liyz*cos(b)*cos(c));

            Jz(4,1) = (2*lixy*cos(a)*cos(b) - 2*lixx*cos(b)*sin(a))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + (2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a)) - 2*liyy*(liyz*sin(b) - liyy*cos(b)*sin(a))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + 2*liyy*cos(a)*cos(b)*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));
            Jz(4,2) = (2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c)) - (2*lixx*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + 2*lixy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*liyy*(liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + liyz*cos(b)*sin(c))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));
            Jz(4,3) = (2*lixy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*lixx*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c)) + (2*lixx*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*lixy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*liyy*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*liyy*(liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - liyz*cos(b)*cos(c))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b));

            Jz(5,1) = (2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a)) - (liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(2*liyz*cos(b) + 2*liyy*sin(a)*sin(b)) - (2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(liyz*sin(b) - liyy*cos(b)*sin(a)) - (2*lixz*cos(b) + 2*lixx*cos(a)*sin(b) + 2*lixy*sin(a)*sin(b))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - 2*lizz^2*cos(b)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + 2*lizz^2*sin(b)*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c));
            Jz(5,2) = (2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c)) + (2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + liyz*cos(b)*sin(c)) + (2*lixx*cos(a)*cos(b)*sin(c) - 2*lixz*sin(b)*sin(c) + 2*lixy*cos(b)*sin(a)*sin(c))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (2*liyz*sin(b)*sin(c) - 2*liyy*cos(b)*sin(a)*sin(c))*(liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - 2*lizz^2*cos(b)*sin(c)*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - 2*lizz^2*sin(b)*sin(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));
            Jz(5,3) = (2*lixy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*lixz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*lixx*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c)) - (2*liyy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*liyz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)))*(liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - liyz*cos(b)*cos(c)) + (2*lixx*cos(a)*cos(b)*cos(c) - 2*lixz*cos(c)*sin(b) + 2*lixy*cos(b)*cos(c)*sin(a))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) - (liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(2*liyz*cos(c)*sin(b) - 2*liyy*cos(b)*cos(c)*sin(a)) - 2*lizz^2*cos(b)*cos(c)*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - 2*lizz^2*cos(c)*sin(b)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));

            Jz(6,1) = - 2*sin(b)*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*lizz^2 + (liyz*sin(b) - liyy*cos(b)*sin(a))*(2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(lixx*cos(a)*cos(b) - lixz*sin(b) + lixy*cos(b)*sin(a));
            Jz(6,2) = (2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + lixz*cos(b)*sin(c)) - (2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + liyz*cos(b)*sin(c)) - (liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(2*liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - 2*liyz*cos(b)*cos(c)) + (2*lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*lixz*cos(b)*cos(c))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*lizz^2*cos(b)*cos(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + 2*lizz^2*cos(b)*sin(c)*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c));
            Jz(6,3) = (2*liyy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - 2*liyz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(liyy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - liyz*cos(b)*cos(c)) - (liyy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + liyz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(2*liyy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + 2*liyz*cos(b)*sin(c)) + (2*lixx*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*lixy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*lixz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(lixx*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - lixy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + lixz*cos(b)*cos(c)) - (2*lixy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*lixx*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*lixz*cos(b)*sin(c))*(lixx*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + lixy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + lixz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + 2*lizz^2*cos(b)*cos(c)*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)) - 2*lizz^2*cos(b)*sin(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));
        end

        function Sigma = compute(obj, pc1, pc2, T_est)
            pts1 = pc1.Location;
            pts2 = pctransform(pc2, affine3d(T_est')).Location;   % transformed cloud
            idx  = knnsearch(pts2, pts1);
            P    = pts1;
            Q    = pts2(idx,:);
            n    = size(P,1);
            
            % KD trees (faster & correct neighborhoods)
            kdtP  = KDTreeSearcher(P);      % full P
            kdtQ2 = KDTreeSearcher(pts2);   % full transformed pts2
            
            Cp_all = zeros(3,3,n);
            Cq_all = zeros(3,3,n);
            
            parfor i = 1:n
                % P-side covariance from full P
                idxP = knnsearch(kdtP,  P(i,:),   'K', obj.K+1);
                Cp   = cov(P(idxP(2:end),:));
                if rank(Cp) < 3, Cp = Cp + 1e-6*eye(3); end
                Cp_all(:,:,i) = obj.regCov(Cp);
            
                % Q-side covariance from full transformed pts2 (NOT only the subset Q)
                idxQ = knnsearch(kdtQ2, Q(i,:),   'K', obj.K+1);
                Cq   = cov(pts2(idxQ(2:end),:));
                if rank(Cq) < 3, Cq = Cq + 1e-6*eye(3); end
                Cq_all(:,:,i) = obj.regCov(Cq);
            end

            % after you’ve computed residuals r_i and W_i at the optimum
            d2J_dX2  = zeros(6,6);
            d2J_dZdX = zeros(6,3*n);
            SSE = 0;
            for i = 1:n
                p  = P(i,:).';
                q  = Q(i,:).';
                Cp = Cp_all(:,:,i);
                Cq = Cq_all(:,:,i);
            
                % Accumulate GN Hessian (force symmetry per term)
                Hi = obj.buildH(p, Cp, q, Cq, eye(4));
                d2J_dX2 = d2J_dX2 + (Hi + Hi.')/2;
                
                % Jz as you had
                d2J_dZdX(:, 3*(i-1)+1 : 3*i) = obj.buildJz(p, Cp, q, Cq, eye(4));
            
                % Consistent SSE (variance factor) using the *same* L
                Lw = obj.infoChol(Cp, Cq, eye(3));     % here R = I because you passed eye(4) above
                e  = Lw * (p - q);
                SSE = SSE + e.'*e;                     % sum of squared whitened residuals
            end
            dof = max(1, 3*n - 6);                     % guard against tiny n
            s2  = SSE / dof;                           % variance factor (can be large if your T_est is wrong)
            B  = zeros(6,6);
            for i = 1:n
                Jzi = d2J_dZdX(:,3*(i-1)+(1:3));
                B   = B + Jzi * (s2*obj.CovZ) * Jzi.';
            end
            H = (d2J_dX2 + d2J_dX2.')/2;   % enforce symmetry

            % Levenberg-style damping (tiny, scale-aware)
            lam = 1e-9 * max(1, trace(H)/6);
            H = H + lam*eye(6);
            
            % Try Cholesky
            [R,p] = chol(H);
            if p>0
                % Eigen repair to nearest SPD (clip tiny/negative eigenvalues)
                [V,D] = eig(H);
                d = diag(D);
                floorVal = 1e-12 * max(1, max(d));
                d = max(d, floorVal);
                H = V*diag(d)*V';
                H = (H+H')/2;
                R = chol(H);   % should succeed now
            end
            
            B = (B + B.')/2;
            Sigma = R \ (R' \ B); 
            Sigma = (Sigma + Sigma.')/2;


            % reorder both rows & cols so params go [a,b,c,x,y,z], as per convention
            perm  = [4 5 6 1 2 3];
            Sigma = Sigma(perm,perm);
        end
    end
end