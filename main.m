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
angle   = pi/6;
axis    = [1; 0; 0];
R_true  = axang2rotm([axis'/norm(axis), angle]);
t_true  = [0.5; -0.3; 0.2];

%% Instantiate generator & estimator
gen = UniformPlaneGenerator(N, n_hat, width, height, "r_true", R_true, "t_true", t_true);
est = CensiCovarianceEstimator;
est.CovZ     = eye(6) * (0.005^2);  % per-point measurement noise
est.Epsilon  = 1e-6;                % finite-difference step

%% Generate clouds and transforms
[pc1, pc2, T_true] = gen.generate();
T_est = T_true;  % here we use the perfect alignment as example

%% Compute predicted covariance
Sigma_est = est.compute(pc1, pc2, T_est);

%% Monte Carlo validation
validator = MonteCarloValidator(gen, est, 'NumSamples', 100);
results   = validator.validate();
Sigma_emp = results.SigmaEmpirical;
MD        = results.Mahalanobis;

%% 1) Compare predicted vs empirical variances (diagonals of Σ)
axes_names = {'ω_x','ω_y','ω_z','t_x','t_y','t_z'};
pred_var   = diag(Sigma_est);
emp_var    = diag(Sigma_emp);

figure;
bar([pred_var, emp_var]);
set(gca, 'XTick', 1:6, 'XTickLabel', axes_names);
legend('Predicted','Empirical','Location','best');
ylabel('Variance');
title('Predicted vs Empirical Covariance Diagonals');

%% 2) Mahalanobis‐distance histogram vs χ²₆ PDF
figure;
histogram(MD, 'Normalization', 'pdf', 'EdgeColor', 'none');
hold on;
x = linspace(0, max(MD), 200);
plot(x, chi2pdf(x,6), 'r-', 'LineWidth', 2);
xlabel('Mahalanobis Distance');
ylabel('PDF');
legend('Empirical','χ^2_6 PDF','Location','best');
title('Mahalanobis Distance Distribution');

%% 3) Q–Q plot of Mahalanobis distances vs theoretical χ²₆
figure;
MDs    = sort(MD);  
p      = ((1:numel(MDs)) - 0.5).' / numel(MDs);   % make p a column
theo_q = chi2inv(p, 6);

plot(theo_q, MDs, 'o');
hold on;

% get the overall max of both for the 45° line
maxv = max([theo_q; MDs]);

plot([0, maxv], [0, maxv], 'r-');
xlabel('Theoretical χ^2_6 Quantiles');
ylabel('Empirical Quantiles');
title('Q–Q Plot of Mahalanobis Distances');
legend('Data','Ideal','Location','best');
