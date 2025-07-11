
%%===================%%
% PointCloudGenerator.m
%%===================%%
classdef (Abstract) PointCloudGenerator
    % Abstract interface: generate two point clouds and the ground-truth transform
    methods (Abstract)
        [pc1, pc2, T_true] = generate(obj)
        % pc1, pc2: pointCloud objects
        % T_true: 4×4 ground-truth transform mapping pc2 into pc1 frame
    end
end