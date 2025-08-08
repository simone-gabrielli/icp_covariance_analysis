clc; clear;

%% Numeric values

% Provide your actual correspondences as an N×6 matrix [pix,piy,piz,qix,qiy,qiz]:
matches = [
    1,1,0, 1,1,1;
    4,4,0, 4,4,1;
    7,7,0, 7,7.1,1
];
N = size(matches,1);

cov_z   = eye(6) * 1e-1;        % measurement noise covariances
theta0  = [0;0;0; 0;0;0];       % current estimate [x;y;z;a;b;c]

%% Exactly like paper " A Closed-form Estimate of 3D ICP Covariance "

% Precompute symbolic Hessian and mixed-derivative templates once
run('point_to_point_first_term.m');  % variables x,y,z,a,b,c,pix,piy,piz,qix,qiy,qiz
H_sym = [
    d2J_dx2,   d2J_dydx, d2J_dzdx, d2J_dadx, d2J_dbdx, d2J_dcdx;...
    d2J_dxdy,  d2J_dy2,  d2J_dzdy, d2J_dady, d2J_dbdy, d2J_dcdy;...
    d2J_dxdz,  d2J_dydz, d2J_dz2,  d2J_dadz, d2J_dbdz, d2J_dcdz;...
    d2J_dxda,  d2J_dyda, d2J_dzda, d2J_da2,  d2J_dbda, d2J_dcda;...
    d2J_dxdb,  d2J_dydb, d2J_dzdb, d2J_dadb,  d2J_db2,  d2J_dbdc;...
    d2J_dxdc,  d2J_dydc, d2J_dzdc, d2J_dadc,  d2J_dcdb, d2J_dc2  ...
];

run('point_to_point_second_term.m'); % defines mixed derivatives
D_sym = [
    d2J_dpix_dx, d2J_dpiy_dx, d2J_dpiz_dx, d2J_dqix_dx, d2J_dqiy_dx, d2J_dqiz_dx;...
    d2J_dpix_dy, d2J_dpiy_dy, d2J_dpiz_dy, d2J_dqix_dy, d2J_dqiy_dy, d2J_dqiz_dy;...
    d2J_dpix_dz, d2J_dpiy_dz, d2J_dpiz_dz, d2J_dqix_dz, d2J_dqiy_dz, d2J_dqiz_dz;...
    d2J_dpix_da, d2J_dpiy_da, d2J_dpiz_da, d2J_dqix_da, d2J_dqiy_da, d2J_dqiz_da;...
    d2J_dpix_db, d2J_dpiy_db, d2J_dpiz_db, d2J_dqix_db, d2J_dqiy_db, d2J_dqiz_db;...
    d2J_dpix_dc, d2J_dpiy_dc, d2J_dpiz_dc, d2J_dqix_dc, d2J_dqiy_dc, d2J_dqiz_dc;...
];
% ---------------------- Numeric assembly starts here ----------------------

% prepare symbolic variable list for subs
symVars = {x,y,z,a,b,c, pix,piy,piz, qix,qiy,qiz};

H_sum_num     = zeros(6,6);
D_sum_num     = [];             % will horiz‐concat all Di blocks

for i = 1:N
    % extract numeric correspondences without overwriting symbols
    pxi = matches(i,1);  pyi = matches(i,2);  pzi = matches(i,3);
    qxi = matches(i,4);  qyi = matches(i,5);  qzi = matches(i,6);

    % build substitution values
    subVals = {
      theta0(1), theta0(2), theta0(3), ...
      theta0(4), theta0(5), theta0(6), ...
      pxi, pyi, pzi, ...
      qxi, qyi, qzi
    };

    % substitute into symbolic templates
    Hi_sym = subs(H_sym,  symVars, subVals);
    Di_sym = subs(D_sym,  symVars, subVals);

    % convert to numeric
    Hi = double(Hi_sym);
    Di = double(Di_sym);

    H_sum_num     = H_sum_num    + Hi;
    D_sum_num     = [D_sum_num, Di];
end

Cov_z_big     = kron(eye(N), cov_z);
Cov_inner_num = D_sum_num * Cov_z_big * D_sum_num.';

Cov_theta_num = H_sum_num \ Cov_inner_num / H_sum_num.';
disp('Numeric covariance of parameters (Cov_theta_num):');
disp(Cov_theta_num);


%% Exact formulation
import sympy.*;
syms x y z a b c pix piy piz qix qiy qiz real

% build the J, g, H, Jz
% Method 1) RPY
% Rz = [ cos(a) -sin(a) 0;
%        sin(a)  cos(a) 0;
%          0        0   1];
% Ry = [ cos(b) 0 sin(b);
%          0    1   0;
%       -sin(b) 0 cos(b)];
% Rx = [ 1   0       0;
%        0 cos(c) -sin(c);
%        0 sin(c)  cos(c)];
% Rmat = Rz*Ry*Rx;
% Method 2) use Taylor‐second‐order parametrization for consistency
S    = [0 -c b; c 0 -a; -b a 0];
Rmat = eye(3) + S + 0.5*(S*S);
p    = [pix; piy; piz];
q    = [qix; qiy; qiz];
Tp   = Rmat*p + [x; y; z];
err  = Tp - q;
J    = err.'*err;
g    = jacobian(J, [x; y; z; a; b; c]).';
H    = hessian(J, [x; y; z; a; b; c]);
Jz   = jacobian(g, [pix; piy; piz; qix; qiy; qiz]);

% preallocate
d2J_dX2  = zeros(6,6);
d2J_dZdX = zeros(6,6*N);

for i = 1:N
    % pack arguments in the same order as subs_vars
    args = { matches(i,1), matches(i,2), matches(i,3), ...
             matches(i,4), matches(i,5), matches(i,6), ...
             theta0(1),   theta0(2),   theta0(3),   ...
             theta0(4),   theta0(5),   theta0(6) };
    Hi  = subs(H,  [pix, piy, piz, qix, qiy, qiz, x, y, z, a, b, c], args);
    Jzi = subs(Jz, [pix, piy, piz, qix, qiy, qiz, x, y, z, a, b, c], args);
    d2J_dX2 = d2J_dX2 + Hi;
    d2J_dZdX(:,6*(i-1)+1:6*i) = Jzi;
end

% form measurement-noise cov and final ICP covariance
bigZ  = kron(eye(N), cov_z);
invHX = inv(d2J_dX2);
Sigma = invHX * d2J_dZdX * bigZ * d2J_dZdX.' * invHX;

disp('Numeric covariance of parameters (Sigma):');
disp(double(Sigma));

%% TEST
% clc; clear;

% % Define point correspondences externally as cell array: each entry = [pix,piy,piz,qix,qiy,qiz]
% % points must be defined before running this script, e.g.:
% points = { [1,2,3,1,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18] };
% n = length(points);

% % Initialize accumulators
% H_sum = sym(zeros(6,6));
% D_sum = sym([]);  % start with empty 6×0 matrix for horizontal concat

% % Precompute symbolic Hessian and mixed-derivative templates once
% run('point_to_point_first_term.m');  % variables x,y,z,a,b,c,pix,piy,piz,qix,qiy,qiz
% H_sym = [
%     d2J_dx2,   d2J_dydx, d2J_dzdx, d2J_dadx, d2J_dbdx, d2J_dcdx;...
%     d2J_dxdy,  d2J_dy2,  d2J_dzdy, d2J_dady, d2J_dbdy, d2J_dcdy;...
%     d2J_dxdz,  d2J_dydz, d2J_dz2,  d2J_dadz, d2J_dbdz, d2J_dcdz;...
%     d2J_dxda,  d2J_dyda, d2J_dzda, d2J_da2,  d2J_dbda, d2J_dcda;...
%     d2J_dxdb,  d2J_dydb, d2J_dzdb, d2J_dadb,  d2J_db2,  d2J_dbdc;...
%     d2J_dxdc,  d2J_dydc, d2J_dzdc, d2J_dadc,  d2J_dcdb, d2J_dc2  ...
% ];

% run('point_to_point_second_term.m'); % defines mixed derivatives
% D_sym = [
%     d2J_dpix_dx, d2J_dpiy_dx, d2J_dpiz_dx, d2J_dqix_dx, d2J_dqiy_dx, d2J_dqiz_dx;...
%     d2J_dpix_dy, d2J_dpiy_dy, d2J_dpiz_dy, d2J_dqix_dy, d2J_dqiy_dy, d2J_dqiz_dy;...
%     d2J_dpix_dz, d2J_dpiy_dz, d2J_dpiz_dz, d2J_dqix_dz, d2J_dqiy_dz, d2J_dqiz_dz;...
%     d2J_dpix_da, d2J_dpiy_da, d2J_dpiz_da, d2J_dqix_da, d2J_dqiy_da, d2J_dqiz_da;...
%     d2J_dpix_db, d2J_dpiy_db, d2J_dpiz_db, d2J_dqix_db, d2J_dqiy_db, d2J_dqiz_db;...
%     d2J_dpix_dc, d2J_dpiy_dc, d2J_dpiz_dc, d2J_dqix_dc, d2J_dqiy_dc, d2J_dqiz_dc;...
% ];

% for k = 1:n
%     vals = points{k};
%     % rename numeric locals so you don't override the symbolic pix,piy,...
%     px = vals(1);  py = vals(2);  pz = vals(3);
%     qx = vals(4);  qy = vals(5);  qz = vals(6);

%     % now subs by the symbolic names (as text) into H_sym/D_sym
%     Hk = subs(H_sym, {'pix','piy','piz','qix','qiy','qiz'}, {px,py,pz,qx,qy,qz});
%     Dk = subs(D_sym, {'pix','piy','piz','qix','qiy','qiz'}, {px,py,pz,qx,qy,qz});

%     H_sum = H_sum + Hk;
%     D_sum = [D_sum, Dk];
% end

% % Assign totals
% H = simplify(H_sum);
% D = simplify(D_sum);

% % Define symbolic covariance of measurement noise z (6x6)
% cov_z = eye(6*n) * sym('cov_z', 'real');

% % Compute final covariance of parameters theta
% Cov_inner = D * cov_z * D.';
% Cov_inner = simplify(Cov_inner);
% H_inv = inv(H);
% X = H_inv * Cov_inner * H_inv;
% Cov_theta = simplify(X);

% % Display result
% disp('Final symbolic covariance of parameters (Cov_theta):');
% disp(Cov_theta);
