import torch
import torch.nn as nn

from mash_shape_diffusion.Model.Layer.point_embed import PointEmbed
from mash_shape_diffusion.Model.Transformer.latent_array import LatentArrayTransformer


class MashNet(torch.nn.Module):
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
    ):
        super().__init__()
        self.n_latents = n_latents
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

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

    def forwardCondition(self, shape_params, pose_params, condition, t):
        pose_embeddings = self.embedPose(pose_params)
        condition = torch.cat([pose_embeddings, condition], dim=1)

        shape_params_noise = self.model(shape_params, t, cond=condition)
        return shape_params_noise

    def forward(self, shape_params, condition_dict, t, condition_drop_prob):
        pose_params = condition_dict['pose_params']
        condition = condition_dict['condition']

        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([shape_params.shape[0]], dtype=torch.long, device=shape_params.device))
        else:
            condition = self.emb_category(condition)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(shape_params.device)
        condition = condition * context_mask

        return self.forwardCondition(shape_params, pose_params, condition, t)
