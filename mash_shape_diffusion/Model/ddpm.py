import torch
import numpy as np
from torch import nn

from mash_shape_diffusion.Method.schedule import ddpm_schedules

class DDPM(nn.Module):
    def __init__(self, nn_model: nn.Module, betas=(1e-4, 0.02), n_T:int=400, device: str='cuda', drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        self.ddpms = ddpm_schedules(betas[0], betas[1], n_T, device)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_fn = nn.MSELoss()
        return

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.ddpms['sqrtab'][_ts, None, None] * x
            + self.ddpms['sqrtmab'][_ts, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.


        # return Loss between added noise, and our predicted noise
        return self.loss_fn(noise, self.nn_model(x_t, c, _ts / self.n_T, self.drop_prob))

    def sample(self, noise, condition_dict, n_sample, guide_w = 0.0, store_num=20, store_last_itr_num=8):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        device = noise.device
        store_skip_num = int(self.n_T/store_num)

        x_i = noise  # x_T ~ N(0, 1), sample initial noise
        c_i = condition_dict

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}        ',end='\r')
            t_is = torch.tensor([i / self.n_T] * n_sample).to(device)

            z = torch.randn(*noise.shape).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps1 = self.nn_model(x_i, c_i, t_is, 0.0)
            eps2 = self.nn_model(x_i, c_i, t_is, 1.0)
            eps = guide_w*eps1 - (1.0 - guide_w)*eps2
            x_i = (
                self.ddpms['oneover_sqrta'][i] * (x_i - eps * self.ddpms['mab_over_sqrtmab'][i])
                + self.ddpms['sqrt_beta_t'][i] * z
            )
            if i%store_skip_num==0 or i==self.n_T or i<=store_last_itr_num:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
