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
        n_heads=8,
        d_head=64,
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

        #self.channels = 6 + self.mask_dim + self.sh_dim
        self.channels = context_dim

        assert context_dim % 4 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, context_dim // 4)
        self.position_embed = PointEmbed(3, d_hidden_embed, context_dim // 4)
        self.mask_embed = PointEmbed(self.mask_dim, d_hidden_embed, context_dim // 4)
        self.sh_embed = PointEmbed(self.sh_dim, d_hidden_embed, context_dim // 4)

        self.category_emb = nn.Embedding(55, context_dim)

        self.model = LatentArrayTransformer(
            in_channels=self.channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )

        self.to_outputs = nn.Linear(self.channels, 6 + self.mask_dim + self.sh_dim)
        return

    def embedMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])
        mask_embeddings = self.mask_embed(mash_params[:, :, 6: 6+ self.mask_dim])
        sh_embeddings = self.sh_embed(mash_params[:, :, 6+self.mask_dim :])

        mash_embeddings = torch.cat(
            [rotation_embeddings, position_embeddings, mask_embeddings, sh_embeddings],
            dim=2,
        )
        return mash_embeddings

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forwardCondition(self, mash_params, condition, t):
        mash_embeddings = self.embedMash(mash_params)

        mash_params_noise = self.model(mash_embeddings, t, cond=condition)
        mash_params_noise = self.to_outputs(mash_params_noise)
        return mash_params_noise

    def forward(self, mash_params, condition, t, condition_drop_prob):
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
        condition = condition * context_mask

        return self.forwardCondition(mash_params, condition, t)
