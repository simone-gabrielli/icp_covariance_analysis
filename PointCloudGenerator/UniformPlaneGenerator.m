%%========================%%
% UniformPlaneGenerator.m
%%========================%%
classdef UniformPlaneGenerator < PointCloudGenerator
    % Generates two point clouds of N points uniformly distributed on a
    % w-by-h rectangle in a plane with normal n_hat, then transforms
    % pc2 = R_true*pc1 + t_true.
    properties
        N (1,1) double
        n_hat (3,1) double
        width (1,1) double
        height (1,1) double
        R_true (3,3) double = eye(3)
        t_true (3,1) double = zeros(3,1)
        z_perturb (1,1) double = 0.0  % Optional perturbation in z direction
    end
    methods
        function obj = UniformPlaneGenerator(N, n_hat, w, h, varargin)
            obj.N = N;
            obj.n_hat = n_hat / norm(n_hat);
            obj.width = w;
            obj.height = h;
            for k = 1:2:numel(varargin)
                switch lower(varargin{k})
                    case 'r_true'
                        obj.R_true = varargin{k+1};
                    case 't_true'
                        obj.t_true = varargin{k+1};
                    case 'z_perturb'
                        % Optional perturbation in z direction
                        obj.z_perturb = varargin{k+1};
                    otherwise
                        error('Unknown parameter %s', varargin{k});
                end
            end
        end

        function [pc1, pc2, T_true] = generate(obj)
            % Build orthonormal basis (u,v) perpendicular to n_hat
            n = obj.n_hat;
            if abs(n(1)) < abs(n(2))
                tmp = [1;0;0];
            else
                tmp = [0;1;0];
            end
            u = cross(n, tmp); u = u / norm(u);
            v = cross(n, u);

            % Sample points
            u_vals1 = (rand(obj.N,1) - 0.5) * obj.width;
            v_vals1 = (rand(obj.N,1) - 0.5) * obj.height;
            % Small perturbation in z (along n_hat)
            z_vals1 = (rand(obj.N,1) - 0.5) * obj.z_perturb; % adjust 0.01 as needed
            pts = u_vals1 .* u' + v_vals1 .* v' + z_vals1 .* n';

            % First cloud
            pc1 = pointCloud(pts);
            % Second cloud

            % Generate second cloud points (same distribution as pc1)
            u_vals2 = (rand(obj.N,1) - 0.5) * obj.width;
            v_vals2 = (rand(obj.N,1) - 0.5) * obj.height;
            z_vals2 = (rand(obj.N,1) - 0.5) * obj.z_perturb;
            pts2_raw = u_vals2 .* u' + v_vals2 .* v' + z_vals2 .* n';

            pts2 = (obj.R_true * pts2_raw')' + repmat(obj.t_true', obj.N, 1);
            pc2 = pointCloud(pts2);

            % Ground-truth transform
            T_true = eye(4);
            T_true(1:3,1:3) = obj.R_true;
            T_true(1:3,4)   = obj.t_true;
        end
    end
end
