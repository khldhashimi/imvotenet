import numpy as np
import torch



calib= np.array([[ 0.9998, -0.0175, -0.0051],
 [ 0.0176,  0.9998,  0.0071],
 [ 0.0050, -0.0072,  0.9999]])
print("calib shape",calib.shape)
calib = torch.tensor(calib)
# change the calib shape from (3,3) to (1,3,3)
calib = calib.unsqueeze(0)
print("calib shape",calib.shape)
calib=calib.transpose(2,1)
print("calib",calib)
#                 XXX_features: float32 Tensor of shape (B,K,3 + input_feature_dim)
#                 XXX_inds: int32 Tensor of shape (B,K)
#                 Indices of the points in the original point-cloud