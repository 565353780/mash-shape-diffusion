import os
import torch
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash

from mash_shape_diffusion.Dataset.mash import MashDataset
from mash_shape_diffusion.Model.ddpm import DDPM
from mash_shape_diffusion.Model.mash_net import MashNet


class Sampler(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.mask_degree = 3
        self.sh_degree = 2
        self.d_hidden_embed = 48
        self.context_dim = 768
        self.n_heads = 1
        self.d_head = 256
        self.depth = 24
        self.device = device

        self.mask_dim = self.mask_degree * 2 + 1
        self.sh_dim = (self.sh_degree + 1) ** 2

        self.model = DDPM(MashNet(n_latents=400, mask_degree=3, sh_degree=2,
                                  d_hidden_embed=48, context_dim=768,n_heads=1,
                                  d_head=256,depth=12),
                          betas=(1e-4, 0.02),
                          n_T=36,
                          device=self.device,
                          drop_prob=0.1
        ).to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)

        HOME = os.environ["HOME"]
        dataset_folder_path_list = [
            HOME + "/Dataset/",
            "/data2/lch/Dataset/",
        ]
        for dataset_folder_path in dataset_folder_path_list:
            if not os.path.exists(dataset_folder_path):
                continue

            self.dataset_root_folder_path = dataset_folder_path
            break

        #TODO: for test only! need to generate pose with another net later
        self.pose_dataset = MashDataset(self.dataset_root_folder_path)
        return

    def toInitialMashModel(self, device: Union[str, None]=None) -> Mash:
        if device is None:
            device = self.device

        mash_model = Mash(
            self.mash_channel,
            self.mask_degree,
            self.sh_degree,
            10,
            400,
            0.4,
            dtype=torch.float32,
            device=device,
        )
        return mash_model

    def loadModel(self, model_file_path, resume_model_only=True):
        if not os.path.exists(model_file_path):
            print("[ERROR][MashSampler::loadModel]")
            print("\t model_file not exist!")
            return False

        model_dict = torch.load(model_file_path, map_location=torch.device(self.device))

        self.model.load_state_dict(model_dict["model"])

        if not resume_model_only:
            # self.optimizer.load_state_dict(model_dict["optimizer"])
            self.step = model_dict["step"]
            self.eval_step = model_dict["eval_step"]
            self.loss_min = model_dict["loss_min"]
            self.eval_loss_min = model_dict["eval_loss_min"]
            self.log_folder_name = model_dict["log_folder_name"]

        print("[INFO][MashSampler::loadModel]")
        print("\t load model success!")
        return True

    @torch.no_grad()
    def sample(
        self,
        sample_num: int,
        category_id: int = 0,
        ) ->list: 
        self.model.eval()

        pose_idxs = np.random.choice(range(len(self.pose_dataset)), sample_num)

        pose_params_list = []
        condition_list = []
        for pose_idx in pose_idxs:
            data = self.pose_dataset[pose_idx]
            mash_params = data['mash_params']
            category_id = data['category_id']
            pose_params = mash_params[:, :6]

            pose_params_list.append(pose_params.unsqueeze(0))
            condition_list.append(category_id)

        pose_params = torch.cat(pose_params_list, dim=0).to(self.device)
        condition = torch.Tensor(condition_list).long().to(self.device)

        condition_dict = {
            'pose_params': pose_params,
            'condition': condition,
        }

        shape_params, middle_shape_params_list = self.model.sample(
            noise=torch.randn(sample_num, self.mash_channel, self.mask_dim + self.sh_dim).to(self.device),
            condition_dict=condition_dict,
            n_sample=sample_num,
            guide_w=1.0,
        )

        pose_params = pose_params.cpu().numpy()

        mash_params_list = []

        for middle_shape_params in middle_shape_params_list:
            mash_params = np.concatenate([pose_params, middle_shape_params], axis=2)
            mash_params_list.append(mash_params)

        return mash_params_list
