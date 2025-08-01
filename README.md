# Point Cloud Alignment Framework

A modular MATLAB framework for:

* Generating pairs of 3D point clouds under a known ground-truth transform.
* Estimating the 6×6 covariance of the alignment error using closed-form or numeric methods.
* Validating the estimated covariance via Monte Carlo sampling and statistical tests.

---

## Prerequisites

* MATLAB R2023a or later
* Computer Vision Toolbox (for `pointCloud` and `pctransform`)

## Installation

1. Clone this repository:

   ```bash
    git clone https://github.com/simone-gabrielli/icp_covariance_analysis
    ```

2. Add "Common" .m files to your MATLAB path.

## Usage

### 1. Generate point clouds

Subclass the abstract `PointCloudGenerator` and override `generate()` to produce `[pc1, pc2, T_true]`. An example is provided in `UniformPlaneGenerator`:

```matlab
gen = UniformPlaneGenerator(2000, [0;0;1], 2.0, 1.5, R_true, t_true);
[pc1, pc2, T_true] = gen.generate();
```

### 2. Estimate covariance

Subclass `CovarianceEstimator` and override `compute(pc1,pc2,T_est)` to return a 6×6 covariance. The provided `CensiCovarianceEstimator` uses the closed-form theorem from Manoj et al. (2015):

```matlab
est = CensiCovarianceEstimator;
est.CovZ = eye(6)*(0.005^2);
est.Epsilon = 1e-6;
Sigma_est = est.compute(pc1, pc2, T_est);
```

### 3. Validate via Monte Carlo

Use `MonteCarloValidator` to tie generator and estimator together and perform statistical tests:

```matlab
validator = MonteCarloValidator(gen, est, 'NumSamples', 3000);
results   = validator.validate();
% results.SigmaEmpirical  - empirical covariance from samples
% results.Mahalanobis    - MDs for χ²₆ tests
```

The demo script `main.m` shows how to plot:

* Diagonal variances: predicted vs empirical
* Histogram of Mahalanobis distances vs χ²₆ PDF
* Q–Q plot for further validation

## Extending the Framework

* **Custom Generators**: create new subclasses of `PointCloudGenerator` for different shapes or real data loading (e.g., `.ply` files).
* **Custom Estimators**: subclass `CovarianceEstimator` to implement analytic Jacobians or other noise models.
* **Visualization & Reporting**: integrate additional plotting or export to logs.