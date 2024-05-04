import torch
import torch.nn as nn

from mash_shape_diffusion.Model.Layer.point_embed import PointEmbed
from mash_shape_diffusion.Model.Transformer.latent_array import LatentArrayTransformer
from mash_shape_diffusion.Method.sample import step_edm_sampler, edm_sampler
from mash_shape_diffusion.Module.stacked_random_generator import StackedRandomGenerator


class MashShapeDiffusion(torch.nn.Module):
    def __init__(
        self,
        n_latents=400,
        mask_degree: int = 3,
        sh_degree: int = 2,
        d_hidden_embed: int = 48,
        context_dim=768,
        n_heads=1,
        d_head=256,
        depth=12,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        use_fp16=False,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.use_fp16 = use_fp16

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        self.channels = self.mask_dim + self.sh_dim

        assert context_dim % 2 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, context_dim // 2)
        self.position_embed = PointEmbed(3, d_hidden_embed, context_dim // 2)

        self.category_emb = nn.Embedding(55, context_dim)

        self.model = LatentArrayTransformer(
            in_channels=self.channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )
        return

    def embedPose(self, pose_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(pose_params[:, :, :3])
        position_embeddings = self.position_embed(pose_params[:, :, 3:])

        pose_embeddings = torch.cat([rotation_embeddings, position_embeddings], dim=2)
        return pose_embeddings

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forwardCondition(self, shape_params, sigma, pose_params, condition, force_fp32=False, **model_kwargs):
        pose_embeddings = self.embedPose(pose_params)
        condition = torch.cat([pose_embeddings, condition], dim=1)

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and shape_params.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(
            (c_in * shape_params).to(dtype), c_noise.flatten(), cond=condition, **model_kwargs
        )
        assert F_x.dtype == dtype
        D_x = c_skip * shape_params + c_out * F_x.to(torch.float32)
        return D_x

    def forward(self, shape_params, sigma, condition_dict, force_fp32=False, **model_kwargs):
        pose_params = condition_dict['pose_params']
        condition = condition_dict['condition']

        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([shape_params.shape[0]], dtype=torch.long, device=shape_params.device))
        else:
            condition = self.emb_category(condition)

        return self.forwardCondition(shape_params, sigma, pose_params, condition, force_fp32, **model_kwargs)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(self, cond, batch_seeds=None, diffuse_steps:int = 18, step_sample: bool=False):
        # print(batch_seeds)
        if cond is not None:
            batch_size, device = *cond.shape, cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        # batch_size, device = *cond.shape, cond.device
        # batch_seeds = torch.arange(batch_size)

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)

        if step_sample:
            return step_edm_sampler(self, latents, cond, randn_like=rnd.randn_like, num_steps=diffuse_steps)

        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like, num_steps=diffuse_steps)
