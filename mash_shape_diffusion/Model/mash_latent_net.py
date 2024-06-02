import torch
import torch.nn as nn

from mash_shape_diffusion.Model.Transformer.latent_array import LatentArrayTransformer


class MashLatentNet(torch.nn.Module):
    def __init__(
        self,
        n_latents: int =400,
        channels: int = 22,
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

        d_hidden = n_heads * d_head
        assert d_hidden % 2 == 0

        self.category_emb = nn.Embedding(55, context_dim)

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            context_dim=context_dim,
        )
        return

    def emb_category(self, class_labels: torch.Tensor) -> torch.Tensor:
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, encoded_mash: torch.Tensor, condition: torch.Tensor, t: float, condition_drop_prob: float = 0.0):
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([encoded_mash.shape[0]], dtype=torch.long, device=encoded_mash.device))
        else:
            condition = self.emb_category(condition)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(encoded_mash.device)
        condition = condition * context_mask

        encoded_mash_noise = self.model(encoded_mash, t, cond=condition)
        return encoded_mash_noise
