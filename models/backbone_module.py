# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048, # number of seeds from whole point cloud
                radius=0.2,#
                nsample=64, # sample 64 points in the area around each seed with radius of 0.2m 
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        # xyz (B, N, 3), features (B, C=1, N)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        # Output: xyz (B, 2048, 3), features (B, C=128, 2048)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features# (B, mlp[-1], npoint) =>(B, 128, 2048)
        # we have 2048 sphere neighborhoods, each of raius 0.2
        # each neighborhood (ball) has 128 features
        # for the next sa layer each ball is considered as a point with 128 feature and the 3D position
        # of each ball is in end_points['sa1_xyz'].

        xyz, features, fps_inds = self.sa2(xyz, features) # xyz in the input is the 3D position of 2048 different balls from previous layer sa1
        # Output: xyz (B, 1024, 3), features (B, C=256, 1024)
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        # end_points['sa2_xyz'] contains the 3D position of the center of the new 1024 neighborhoods 
        end_points['sa2_features'] = features # (B, mlp[-1], npoint) =>(B, 256, 1024)
        # now we have 1024 sphere neighborhoods in another stage, each of raius 0.4
        # each neighborhood (ball) has 256 features
        # for the next sa layer each ball is considered again as a point with 256 features and the 3D position
        # of each ball is in end_points['sa2_xyz'].

        xyz, features, fps_inds = self.sa3(xyz, features) # xyz in the input is the 3D position of 1024 different balls from previous layer sa2
        # Output: xyz (B, 512, 3), features (B, C=256, 512)
        end_points['sa3_xyz'] = xyz
        # end_points['sa3_xyz'] contains the 3D position of the center of the new 512 neighborhoods 
        end_points['sa3_features'] = features # (B, mlp[-1], npoint) =>(B, 256, 512)
        # now we have 512 sphere neighborhoods in a higher stage, each of raius 0.8
        # each neighborhood (ball) has 256 features
        # for the next sa layer each ball is considered again as a point with 256 features and the 3D position
        # of each ball is in end_points['sa3_xyz'].

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        # Output: xyz (B, 256, 3), features (B, C=256, 256)
        end_points['sa4_xyz'] = xyz
        # end_points['sa4_xyz'] contains the 3D position of the center of the new 256 neighborhoods 
        end_points['sa4_features'] = features# (B, mlp[-1], npoint) =>(B, 256, 256)

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features']) 
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features) # (B, mlp[-1]=256, n) tensor of the features of the unknown features
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
