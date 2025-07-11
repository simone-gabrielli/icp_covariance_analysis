% main.m
% Demonstration script: generate two point clouds, estimate covariance, validate via Monte Carlo,
% and plot how well the predicted covariance matches the true rototranslational errors.

addpath("PointCloudGenerator\")
addpath("CovarianceEstimator\")
addpath("CovarianceValidator\")

clear; close all; clc;
rng(0);

%% Parameters
N       = 100;                           % Number of points per cloud
n_hat   = [0; 0; 1];                      % Plane normal
width   = 2.0;                            % Plane width
height  = 1.5;                            % Plane height

% Ground-truth transform (rotate about x by 30°, translate by [0.5; -0.3; 0.2])
angle   = 0;
axis    = [1; 0; 0];
R_true  = axang2rotm([axis'/norm(axis), angle]);
t_true  = [0.; 0.; 0.];

%% Instantiate generator & estimator
gen = UniformPlaneGenerator(N, n_hat, width, height, "r_true", R_true, "t_true", t_true);
est = CensiCovarianceEstimator;
est.CovZ     = eye(6) * (0.005^2);  % per-point measurement noise
est.Epsilon  = 1e-6;                % finite-difference step
%% Monte Carlo validation
validator = MonteCarloValidator(gen, est, 'NumSamples', 3, "sigma_sq_rot", [0.0 ,0.0 ,0.0], "sigma_sq_trans", [1.0, 1.0, 0.0]);
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
    scatter(abs(perturbations(:,k)), var(:,k), 25, 'filled');
    xlabel(['|Actual ', axes_names{k}, '|']);
    ylabel(['Estimated Covariance ', axes_names{k}]);
    title(['Covariance vs |Actual ', axes_names{k}, '|']);
    grid on;
end
for k = 4:6
    subplot(3,2,2*(k-3)); % Right column: translations
    scatter(abs(perturbations(:,k)), var(:,k), 25, 'filled');
    xlabel(['|Actual ', axes_names{k}, '|']);
    ylabel(['Estimated Covariance ', axes_names{k}]);
    title(['Covariance vs |Actual ', axes_names{k}, '|']);
    grid on;
end
sgtitle('Actual Transformations vs Estimated Covariances');