from copy import deepcopy
import sys
sys.path.append('../ma-sh')

import torch
import open3d as o3d

from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud


class EDMLoss:
    def __init__(self, P_mean=-6.0, P_std=2.0, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, condition_dict, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(inputs) * sigma

        noise_data = inputs + n
        return noise_data

def test():
    edm_loss = EDMLoss()

    mash_file_path = '/home/chli/Dataset/MashV3/ShapeNet/03001627/46bd3baefe788d166c05d60b45815.npy'

    mash = Mash.fromParamsFile(mash_file_path, device='cpu')

    # gt_points = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))
    # o3d.io.write_point_cloud("./output/test_noise_gt.ply", gt_points, write_ascii=True)

    mask_params = mash.mask_params
    sh_params = mash.sh_params

    shape_params = torch.hstack([mask_params, sh_params]).unsqueeze(0)

    print(torch.min(shape_params))
    print(torch.max(shape_params))
    print(torch.mean(shape_params))

    noise_shape_params = deepcopy(shape_params)
    for _ in range(36):
        noise_shape_params = edm_loss(None, noise_shape_params, None)

    noise_shape_params = noise_shape_params.squeeze(0)

    print(torch.min(noise_shape_params))
    print(torch.max(noise_shape_params))
    print(torch.mean(noise_shape_params))

    sh2d = mask_params.shape[1]
    noise_mask_params = noise_shape_params[:, :sh2d]
    noise_sh_params = noise_shape_params[:, sh2d:]

    mash.loadParams(mask_params=noise_mask_params, sh_params=noise_sh_params)

    noise_points = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))

    o3d.io.write_point_cloud("./output/test_noise_noise.ply", noise_points, write_ascii=True)
    return True
