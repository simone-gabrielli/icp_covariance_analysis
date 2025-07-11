classdef (Abstract) CovarianceEstimator
    % Abstract interface: compute 6×6 covariance for alignment error
    methods (Abstract)
        Sigma = compute(obj, pc1, pc2, T_est)
        % pc1, pc2: pointCloud objects
        % T_est:   4×4 transform aligning pc2 to pc1 (e.g. from ICP)
        % Sigma: 6×6 estimated covariance in se(3) tangent coordinates
    end
end