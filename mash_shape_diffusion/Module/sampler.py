import os
import torch
from typing import Union

from ma_sh.Model.mash import Mash

from mash_autoencoder.Module.detector import Detector

from mash_shape_diffusion.Model.ddpm import DDPM
from mash_shape_diffusion.Model.mash_net import MashNet
from mash_shape_diffusion.Model.mash_ssm import MashSSM
from mash_shape_diffusion.Model.mash_latent_net import MashLatentNet


class Sampler(object):
    def __init__(
        self, model_file_path: Union[str, None] = None, ae_model_file_path: Union[str, None]=None, device: str = "cpu"
    ) -> None:
        self.mash_channel = 400
        self.encoded_mash_channel = 10
        self.mask_degree = 3
        self.sh_degree = 2
        self.d_hidden_embed = 48
        self.context_dim = 768
        self.n_heads = 8
        self.d_head = 64
        self.depth = 24
        self.device = device

        model_id = 2
        if model_id == 1:
            base_model = MashNet(n_latents=self.mash_channel, mask_degree=self.mask_degree, sh_degree=self.sh_degree,
                                    d_hidden_embed=self.d_hidden_embed, context_dim=self.context_dim,n_heads=self.n_heads,
                                    d_head=self.d_head,depth=self.depth)
        elif model_id == 2:
            base_model = MashLatentNet(n_latents=self.mash_channel, channels=self.encoded_mash_channel,
                                    context_dim=self.context_dim,n_heads=self.n_heads,
                                    d_head=self.d_head,depth=self.depth)
        elif model_id == 3:
            #base_model = MashSSM().to(self.device)
            pass

        self.model = DDPM(base_model,
                          betas=(1e-4, 0.02),
                          n_T=400,
                          device=self.device,
        ).to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.detector = Detector(ae_model_file_path, device='cuda')
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
        ) -> list: 
        self.model.eval()

        condition = torch.ones([sample_num]).long().to(self.device) * category_id

        latents, middle_latents_array = self.model.sample(
            noise=torch.randn(sample_num, self.mash_channel, self.encoded_mash_channel).to(self.device),
            condition=condition,
            n_sample=sample_num,
            guide_w=1.0,
        )

        mash_params_list = self.detector.decodeLatent(latents.to('cuda')).to(self.device)

        mash_params_list = []

        for middle_latents in middle_latents_array:
            middle_mash_params = self.detector.decodeLatent(torch.from_numpy(middle_latents).type(latents.dtype).to('cuda')).to(self.device)
            mash_params_list.append(middle_mash_params)

        return mash_params_list
