# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

def append_img_feat(img_feat_list, end_points):
    """
    This function takes the raw 2D information gathered by ImageFeatureModule and
    fuses it with the 3D features from the PointNet++ backbone.
    It computes and appends geometric, semantic, and texture cues.
    
    Args:
        img_feat_list (list): A list of tensors from ImageFeatureModule. Each tensor
                              contains raw 2D vote information for each 3D seed point.
        end_points (dict): A dictionary from the model containing intermediate tensors,
                           including 3D features, coordinates, and calibration data.
    Returns:
        xyz (torch.Tensor): The coordinates of the seed points.
        features (torch.Tensor): The final, fused feature tensor combining 3D and 2D cues.
        seed_inds (torch.Tensor): The indices of the seed points.
    """
    feat_list = []
    xyz_list = []
    seed_inds_list = []

    # Unpack necessary tensors from the end_points dictionary
    seed_inds = end_points['fp2_inds']           # Indices of the seed points from the original point cloud
    xyz = end_points['fp2_xyz']                  # (B, num_seed, 3) XYZ coordinates of seed points
    batch_size = xyz.shape[0]
    num_seed = xyz.shape[1]
    fp2_features = end_points['fp2_features']    # (B, C, num_seed) 3D features from the PointNet++ backbone
    semantic_cues = end_points['cls_score_feats']# (B, K, C_sem) Pre-computed semantic features for each 2D box instance
    texture_cues = end_points['full_img_1d']     # (B, H*W*C_tex) Flattened texture features from the ResNet backbone
    calib_Rtilt = end_points['calib_Rtilt']      # (B, 3, 3) Rotation matrix to align depth camera with gravity

    # Iterate through each potential 2D vote (e.g., up to 3 votes per pixel)
    for img_feat in img_feat_list:
        # == 1. COMPUTE GEOMETRIC CUES ==
        # This section calculates features based on the geometric relationship between
        # the 3D point and the 2D vote's location.

        # Create a 2D vector representing the 2D vote location and rotate it into the 3D camera's coordinate system
        img_feat_xyz_camera = torch.cat((img_feat[:,:,0:2], torch.zeros((batch_size, num_seed, 1)).cuda()), -1)
        img_feat_xyz_depth = img_feat_xyz_camera[:,:,[0,2,1]]
        img_feat_xyz_depth[:,:,2] *= -1
        # Align the vector with the upright depth coordinate system using the Rtilt matrix
        img_feat_xyz_upright_depth = torch.matmul(calib_Rtilt, img_feat_xyz_depth.transpose(2,1))
        
        # Calculate the "ray angle": a normalized vector from the camera origin through the 3D seed point
        ray_angle = xyz + img_feat_xyz_upright_depth.transpose(2,1)
        ray_angle /= torch.sqrt(torch.sum(ray_angle**2, -1)+1e-6).unsqueeze(-1) # Normalize the vector
        img_mask = img_feat[:,:,-1].unsqueeze(-1) # Mask to zero out points that don't have a valid 2D vote
        ray_angle *= img_mask

        # Compute C'': A sophisticated geometric feature representing the projected offset
        new_img_feat_xz_upright = torch.zeros((batch_size, num_seed, 2)).cuda()
        new_img_feat_xz_upright[:,:,0] = ray_angle[:,:,0]/(ray_angle[:,:,1]+1e-6) * xyz[:,:,1] - xyz[:,:,0]
        new_img_feat_xz_upright[:,:,1] = ray_angle[:,:,2]/(ray_angle[:,:,1]+1e-6) * xyz[:,:,1] - xyz[:,:,2]
        new_img_feat_xz_upright *= img_mask

        # Prepare a list of features to be concatenated. Start with the original 3D features.
        # Note: Features are transposed to (B, C, num_seed) for 1D convolution
        sub_feat_list = [fp2_features, new_img_feat_xz_upright.transpose(2,1), ray_angle.transpose(2,1)]

        # == 2. APPEND SEMANTIC CUES ==
        # Look up the semantic class features corresponding to the 2D detection instance.
        img_feat_sem = img_feat[:,:,4].long() # Get the instance ID for each vote
        # Use the instance ID to 'gather' the corresponding semantic feature vector
        img_feat_sem = torch.gather(semantic_cues, 1, img_feat_sem.unsqueeze(-1).repeat(1,1,semantic_cues.shape[-1])).transpose(2, 1)
        sub_feat_list.append(img_feat_sem)

        # == 3. APPEND TEXTURE CUES ==
        # Look up the visual texture features from the ResNet feature map.
        img_feat_tex = img_feat[:,:,2:4].long() # Get the (u, v) coordinates for the lookup
        # Calculate the 1D index into the flattened feature map
        img_feat_tex = (img_feat_tex[:,:,0] + img_feat_tex[:,:,1] * end_points['full_img_width'].unsqueeze(-1))*3
        # Gather the 3 channels of texture features
        ch0 = torch.gather(texture_cues, 1, img_feat_tex).unsqueeze(1)
        ch1 = torch.gather(texture_cues, 1, img_feat_tex+1).unsqueeze(1)
        ch2 = torch.gather(texture_cues, 1, img_feat_tex+2).unsqueeze(1)
        img_feat_tex = torch.cat((ch0,ch1,ch2), 1)
        sub_feat_list.append(img_feat_tex)

        # == 4. FINALIZE FUSION ==
        # Concatenate all features (3D, geometric, semantic, texture) into a single feature vector
        feat = torch.cat(sub_feat_list, 1)
        feat_list.append(feat)
        xyz_list.append(xyz)
        seed_inds_list.append(seed_inds)

    # If multiple votes were processed, concatenate their results
    features = torch.cat(feat_list, -1)
    xyz = torch.cat(xyz_list, 1)
    seed_inds = torch.cat(seed_inds_list, 1)
    return xyz, features, seed_inds


class ImageFeatureModule(nn.Module):
    """
    This module is the first step in the fusion process. Its job is to project 3D seed points
    onto the 2D image plane and, for each point, gather raw "vote" information from a
    pre-computed map. This map contains data about which 2D detected objects cover each pixel.
    """
    def __init__(self, max_imvote_per_pixel=3):
        super().__init__()
        self.max_imvote_per_pixel = max_imvote_per_pixel # Max 2D objects a single pixel can vote for
        self.vote_dims = 1+self.max_imvote_per_pixel*4   # Dimension of stored info per pixel

    def forward(self, end_points):
        # == 1. PROJECT 3D POINTS TO 2D IMAGE PLANE ==
        # Transform 3D seed point coordinates from world/depth space to camera image space (u,v coordinates)
        xyz2 = torch.matmul(end_points['calib_Rtilt'].transpose(2,1), (1/(end_points['scale']**2)).unsqueeze(-1).unsqueeze(-1)*end_points['fp2_xyz'].transpose(2,1))
        xyz2 = xyz2.transpose(2,1)
        xyz2[:,:,[0,1,2]] = xyz2[:,:,[0,2,1]]
        xyz2[:,:,1] *= -1
        end_points['xyz_camera_coord'] = xyz2
        # Apply camera intrinsic matrix K to get pixel coordinates
        uv = torch.matmul(xyz2, end_points['calib_K'].transpose(2,1))
        uv[:,:,0] /= uv[:,:,2] # Perspective division
        uv[:,:,1] /= uv[:,:,2]

        # Round to get integer pixel indices
        u = (uv[:,:,0]-1).round()
        v = (uv[:,:,1]-1).round()

        # == 2. GATHER 2D VOTE INFORMATION ==
        # Look up pre-computed 2D vote data using the pixel indices
        full_img_votes_1d = end_points['full_img_votes_1d'] # The large, flattened tensor of 2D vote data
        # Calculate the starting index for each seed point in the flattened tensor
        idx_beg = (u.float() + v.float() * end_points['full_img_width'].unsqueeze(-1).float())*self.vote_dims
        idx_beg = idx_beg.long()
        
        # Get the number of valid votes for this pixel
        seed_gt_votes_cnt = torch.gather(full_img_votes_1d, 1, idx_beg)
        
        img_feat_list = []
        batch_size = xyz2.shape[0]
        num_seed = xyz2.shape[1]
        
        # For each of the possible votes (e.g., up to 3)
        for i in range(self.max_imvote_per_pixel):
            # Gather the 2D coordinates of the vote (e.g., center of the 2D box)
            vote_i_0 = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4)
            vote_i_1 = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4+1)
            seed_gt_votes_i = torch.cat((vote_i_0.unsqueeze(-1), vote_i_1.unsqueeze(-1)), -1)
            # Create a mask indicating if this vote is valid for the point
            seed_gt_votes_mask_i = (seed_gt_votes_cnt > i).float()

            # Scale the 2D vote coordinates by the point's depth
            seed_gt_votes_i *= xyz2[:,:,2].unsqueeze(-1)
            seed_gt_votes_i /= end_points['calib_K'][:,0,0].unsqueeze(-1).unsqueeze(-1)

            # Gather the instance ID of the 2D object this vote belongs to
            ins_id = torch.gather(full_img_votes_1d, 1, idx_beg+1+i*4+3).unsqueeze(-1)
            # Concatenate all gathered info: [2D_vote_xy, pixel_uv, instance_id, valid_mask]
            img_feat_list_i = torch.cat((seed_gt_votes_i, u.unsqueeze(-1), v.unsqueeze(-1), ins_id, seed_gt_votes_mask_i.unsqueeze(-1)), -1)
            img_feat_list.append(img_feat_list_i)

        return img_feat_list


class ImageMLPModule(nn.Module):
    """
    A simple MLP (Multi-Layer Perceptron) that processes the final fused feature vector.
    It learns to weigh the different cues (3D, geometric, semantic, texture) and
    produces a refined feature representation ready for the next stage (voting).
    """
    def __init__(self, input_dim, image_hidden_dim=256):
        super().__init__()
        # 1D Convolutions are used to act as shared MLPs applied to each point feature
        self.img_feat_conv1 = torch.nn.Conv1d(input_dim, image_hidden_dim, 1)
        self.img_feat_conv2 = torch.nn.Conv1d(image_hidden_dim, image_hidden_dim, 1)
        self.img_feat_bn1 = torch.nn.BatchNorm1d(image_hidden_dim)
        self.img_feat_bn2 = torch.nn.BatchNorm1d(image_hidden_dim)

    def forward(self, img_features):
        """
        Args:
            img_features (torch.Tensor): The concatenated feature tensor from append_img_feat.
        Returns:
            img_features (torch.Tensor): The processed and refined features.
        """
        img_features = F.relu(self.img_feat_bn1(self.img_feat_conv1(img_features)))
        img_features = F.relu(self.img_feat_bn2(self.img_feat_conv2(img_features)))

        return img_features