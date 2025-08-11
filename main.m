% main.m
% Demonstration script: generate two point clouds, estimate covariance, validate via Monte Carlo,
% and plot how well the predicted covariance matches the true rototranslational errors.

addpath("PointCloudGenerator\")
addpath("CovarianceEstimator\")
addpath("CovarianceValidator\")

clear; close all; clc;
rng(0);

%% Parameters
N       = 50;                           % Number of points per cloud
n_hat   = [0; 0; 1];                      % Plane normal
width   = 1.0;                            % Plane width
height  = 1.0;                            % Plane height

% Ground-truth transform
angle   = 0;
axis    = [1; 0; 0];
R_true  = axang2rotm([axis'/norm(axis), angle]);
t_true  = [0.; 0.; 0.];

%% Instantiate generator & estimator
gen = RandomPlaneGenerator(N, n_hat, width, height, "r_true", R_true, "t_true", t_true, "z_perturb", 0.00001);
est = CensiEstimatorS2MGICP;
est.CovZ     = eye(3) * (0.001);  % per-point measurement noise S2M
%est.CovZ     = eye(6) * (0.1);  % per-point measurement noise S2S
%% Monte Carlo validation
validator = MonteCarloValidator(gen, est, 'NumSamples', 50, "sigma_sq_rot", [0.0 ,0.0 ,0.0], "sigma_sq_trans", [0., 0., .1]);
results   = validator.validate();

%% 1) Compare predicted vs empirical variances (diagonals of Σ)
axes_names = {'ω_x','ω_y','ω_z','t_x','t_y','t_z'};
% Collect all diagonal elements from each covariance matrix
num_results = numel(results.Covariances);
var = zeros(num_results, 6);
for i = 1:num_results
    var(i, :) = diag(results.Covariances{i});
end
perturbations = zeros(num_results, 6);
for i = 1:num_results
    perturbations(i, :) = SE3Utils.se3Log(results.Transforms{i});
end

% Plot variances and perturbations
figure;
for k = 1:3
    subplot(3,2,2*k-1); % Left column: angles
    scatter(perturbations(:,k), var(:,k), 25, 'filled');
    xlabel(['|Actual ', axes_names{k}, '|']);
    ylabel(['Estimated Covariance ', axes_names{k}]);
    title(['Covariance vs |Actual ', axes_names{k}, '|']);
    grid on;
end
for k = 4:6
    subplot(3,2,2*(k-3)); % Right column: translations
    scatter(perturbations(:,k), var(:,k), 25, 'filled');
    xlabel(['|Actual ', axes_names{k}, '|']);
    ylabel(['Estimated Covariance ', axes_names{k}]);
    title(['Covariance vs |Actual ', axes_names{k}, '|']);
    grid on;
end
sgtitle('Actual Transformations vs Estimated Covariances');

% Compute direction of displacement for each sample
% displacement_dirs = zeros(num_results, 6);
% displacement_mags = zeros(num_results, 1);
% projected_vars = zeros(num_results, 1);

% for i = 1:num_results
%     xi = perturbations(i, :)'; % se(3) vector
%     dir = xi / norm(xi);       % direction of displacement
%     displacement_dirs(i, :) = dir';
%     displacement_mags(i) = norm(xi);
%     Sigma = results.Covariances{i};
%     % Project covariance onto displacement direction
%     projected_vars(i) = dir' * Sigma * dir;
% end

% % Plot projected variance vs. displacement magnitude
% figure;
% scatter(displacement_mags, projected_vars, 40, 'filled');
% xlabel('Displacement Magnitude (||\xi||)');
% ylabel('Estimated Covariance Along Displacement');
% title('Covariance Projected Along Actual Displacement Direction');
% grid on;

% For each sample, find the axis with maximum translation displacement and compare its covariance
max_disp_axis = zeros(num_results,1); % 1=x, 2=y, 3=z
max_disp_value = zeros(num_results,1);
max_disp_cov = zeros(num_results,1);

for i = 1:num_results
    trans_disp = abs(perturbations(i,4:6)); % t_x, t_y, t_z
    [max_disp_value(i), max_disp_axis(i)] = max(trans_disp);
    max_disp_cov(i) = var(i, 3 + max_disp_axis(i)); % covariance for t_x/t_y/t_z
end

% Plot: max displacement vs. covariance along that axis
figure;
scatter(max_disp_value, max_disp_cov, 40, max_disp_axis, 'filled');
xlabel('Maximum Translation Displacement (|t_{max}|)');
ylabel('Estimated Covariance Along Max Displacement Axis');
title('Covariance vs. Maximum Translation Displacement Axis');
colorbar;
colormap([1 0 0; 0 1 0; 0 0 1]); % Red=x, Green=y, Blue=z
caxis([1 3]);
grid on;

% Optionally, plot for each axis separately
figure;
for k = 1:3
    subplot(1,3,k);
    scatter(perturbations(:,3+k), var(:,3+k), 40, 'filled');
    xlabel(['|t_', axes_names{k+3}, '|']);
    ylabel(['Covariance ', axes_names{k+3}]);
    title(['Covariance vs |t_', axes_names{k+3}, '|']);
    grid on;
end
sgtitle('Covariance vs Translation Displacement for Each Axis');