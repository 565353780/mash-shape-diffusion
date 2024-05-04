import torch


class EDMLoss:
    # def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, condition, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)

        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(inputs) * sigma

        D_yn = net(inputs + n, sigma, condition)
        loss = weight * ((D_yn - inputs) ** 2)
        return loss.mean()
