import torch
import open3d as o3d
from tqdm import trange
from copy import deepcopy

import sys
sys.path.append('../ma-sh')
from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud


from mash_shape_diffusion.Method.schedule import ddpm_schedules

def test():
    mash_file_path = '/home/chli/Dataset/MashV3/ShapeNet/03001627/46bd3baefe788d166c05d60b45815.npy'
    diffu_steps = 400
    ddpm_sche = ddpm_schedules(1e-4, 0.02, diffu_steps)
    save_sample_num = 20

    mash = Mash.fromParamsFile(mash_file_path, 10, 400, 0.4, device='cpu')
    mask_params = mash.mask_params
    sh_params = mash.sh_params

    shape_params = torch.hstack([mask_params, sh_params]).unsqueeze(0)

    sqrtab = ddpm_sche['sqrtab']
    sqrtmab = ddpm_sche['sqrtmab']

    print(torch.min(sqrtab))
    print(torch.mean(sqrtab))
    print(torch.max(sqrtab))
    print(torch.min(sqrtmab))
    print(torch.mean(sqrtmab))
    print(torch.max(sqrtmab))
    exit()

    for i in trange(save_sample_num):
        current_t = int(i / (save_sample_num - 1) * diffu_steps)

        noise = torch.randn_like(shape_params)

        print(torch.min(shape_params))
        print(torch.max(shape_params))
        print(torch.mean(shape_params))

        noise_shape_params = deepcopy(shape_params)
        noise_shape_params = sqrtab[current_t, None, None] * noise_shape_params + sqrtmab[current_t, None, None] * noise

        noise_shape_params = noise_shape_params.squeeze(0)

        print(torch.min(noise_shape_params))
        print(torch.max(noise_shape_params))
        print(torch.mean(noise_shape_params))

        sh2d = mask_params.shape[1]
        noise_mask_params = noise_shape_params[:, :sh2d]
        noise_sh_params = noise_shape_params[:, sh2d:]

        mash.loadParams(mask_params=noise_mask_params, sh_params=noise_sh_params)

        noise_points = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))

        o3d.io.write_point_cloud('./output/noise_mash_' + str(current_t) + '.ply', noise_points, write_ascii=True)
    return True
