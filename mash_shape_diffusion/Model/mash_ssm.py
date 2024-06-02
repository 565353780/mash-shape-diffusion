import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from mash_shape_diffusion.Model.mamba_block import create_block, init_weights
from mash_shape_diffusion.Model.positional_embedding import PositionalEmbedding
from mash_shape_diffusion.Model.Layer.pre_norm import PreNorm
from mash_shape_diffusion.Model.Layer.feed_forward import FeedForward
from mash_shape_diffusion.Model.Layer.attention import Attention
from mash_shape_diffusion.Model.Layer.point_embed import PointEmbed


class MashSSM(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        d_hidden: int = 768,
        d_hidden_embed: int = 48,
        d_cond: int = 768,
        d_t: int = 256,
        n_layer: int = 12,
        n_cross: int = 1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        device="cuda:0",
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        assert d_hidden % 4 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.position_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.mask_embed = PointEmbed(self.mask_dim, d_hidden_embed, d_hidden // 4)
        self.sh_embed = PointEmbed(self.sh_dim, d_hidden_embed, d_hidden // 4)

        self.map_noise = PositionalEmbedding(d_t)

        self.map_layer0 = nn.Linear(in_features=d_t, out_features=d_hidden)
        self.map_layer1 = nn.Linear(in_features=d_hidden, out_features=d_hidden)

        self.category_emb = nn.Embedding(55, d_cond)


        self.layers = nn.ModuleList(
            [
                create_block(
                    d_hidden,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_hidden, eps=norm_epsilon, **factory_kwargs
        )


        self.decoder_cross_attn = PreNorm(
            d_hidden,
            Attention(d_hidden, d_hidden, heads=n_cross, dim_head=d_hidden),
            context_dim=d_hidden,
        )
        self.decoder_ff = PreNorm(d_hidden, FeedForward(d_hidden))

        # remove t and condition channel
        self.fuse_channel = nn.Linear(402, 400)
        self.to_outputs = nn.Linear(d_hidden, 6 + self.mask_dim + self.sh_dim)

        self.apply(
            partial(
                init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
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

        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        hidden_states = torch.cat([t_emb, condition, mash_embeddings], dim=1)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        latents = self.decoder_cross_attn(hidden_states, context=condition)

        latents = latents + self.decoder_ff(latents)

        latents = self.fuse_channel(latents.permute(0, 2, 1)).permute(0, 2, 1)
        shape_params_noise = self.to_outputs(latents).squeeze(-1)
        return shape_params_noise

    def forward(self, mash_params, condition, t, condition_drop_prob):
        if condition.dtype == torch.float32:
            condition = condition + 0.0 * self.emb_category(torch.zeros([mash_params.shape[0]], dtype=torch.long, device=mash_params.device))
        else:
            condition = self.emb_category(condition)

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(condition)-condition_drop_prob).to(mash_params.device)
        condition = condition * context_mask

        return self.forwardCondition(mash_params, condition, t)
