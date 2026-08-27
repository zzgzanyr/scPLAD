from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def make_beta_schedule(schedule_name: str, steps: int):
    if schedule_name == "linear":
        scale = 1000 / steps
        return np.linspace(scale * 0.0001, scale * 0.02, steps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise ValueError(f"Unknown schedule: {schedule_name}")


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PatchAutoEncoder(nn.Module):
    """PatchAE used as the frozen latent chart."""

    def __init__(
        self,
        gene_size: int,
        patch_size: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        latent_norm: str = "none",
    ):
        super().__init__()
        self.gene_size = int(gene_size)
        self.patch_size = int(patch_size)
        self.num_patches = math.ceil(self.gene_size / self.patch_size)
        self.padded_gene_size = self.num_patches * self.patch_size
        self.pad_genes = self.padded_gene_size - self.gene_size
        self.latent_dim = int(latent_dim)
        self.latent_norm = latent_norm
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.patch_size),
            nn.Linear(self.patch_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, latent_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.patch_size),
        )

    def encode(self, x):
        bsz = x.shape[0]
        if self.pad_genes:
            x = torch.nn.functional.pad(x, (0, self.pad_genes))
        h = x.view(bsz, self.num_patches, self.patch_size)
        z = self.encoder(h) + self.pos_embed
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        if self.latent_norm == "global_l2":
            flat = z.reshape(bsz, -1)
            z = torch.nn.functional.normalize(flat, p=2, dim=1).reshape_as(z)
        elif self.latent_norm == "token_l2":
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z

    def decode(self, z):
        patches = self.decoder(z)
        recon = patches.reshape(z.shape[0], self.padded_gene_size)
        return recon[:, : self.gene_size]

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, cond_tokens):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN(cond_tokens).chunk(6, dim=-1)
        attn_in = self.modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_attn * self.dropout(attn_out)
        mlp_in = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.dropout(self.mlp(mlp_in))
        return x


class PlainTransformerBlock(nn.Module):
    """Transformer block with no conditional modulation."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ControlAnchoredDisplacementDenoiser(nn.Module):
    """
    Denoise displacement tokens u = z_y - control_anchor.

    The decoder never receives u directly. During generation, the sampled
    displacement is added back to the control anchor:

        z_hat_y = control_anchor + u_hat
    """

    def __init__(
        self,
        num_patches: int,
        latent_dim: int,
        condition_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(latent_dim, hidden_dim)
        self.out_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(hidden_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_feature_dim),
            nn.Linear(condition_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(latent_dim, hidden_dim)
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        ff_dim = int(hidden_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode_condition_stream(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h = self.in_proj(u_t) + self.pos_embed
        time_tokens = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim).to(h.dtype))[:, None, :]
        condition_tokens = self.condition_encoder(condition_features.to(dtype=h.dtype))[:, None, :]
        context_tokens = self.context_encoder(context_features.to(dtype=h.dtype))[:, None, :]
        anchor_tokens = self.anchor_proj(control_anchor.to(dtype=h.dtype))
        condition_tokens = condition_tokens.expand(-1, h.shape[1], -1)
        context_tokens = context_tokens.expand(-1, h.shape[1], -1)
        time_tokens = time_tokens.expand(-1, h.shape[1], -1)
        return h, time_tokens, condition_tokens, context_tokens, anchor_tokens

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, context_tokens, anchor_tokens = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        interaction_tokens = self.interaction_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )
        cond = time_tokens + condition_tokens + context_tokens + anchor_tokens + interaction_tokens
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h))


class SimpleConditionDisplacementDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    Minimal conditioning ablation: denoise with timestep + GO only.

    The control anchor is used only after sampling:

        z_hat_y = control_anchor + u_hat
    """

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, _, _ = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        cond = time_tokens + condition_tokens
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h))


class PooledAnchorConditionDisplacementDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    Lightweight anchor conditioning: use timestep + GO + one pooled anchor token
    broadcast back to all patches, without patch-wise anchor injection.
    """

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, _, anchor_tokens = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        pooled_anchor = anchor_tokens.mean(dim=1, keepdim=True).expand(-1, h.shape[1], -1)
        cond = time_tokens + condition_tokens + pooled_anchor
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h))


class PrefixInteractionConditionDisplacementDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    One-shot GO x control interaction.

    A small set of GO-derived query tokens attends to the control-anchor patch
    tokens once at the input stage. The resulting interaction prefix tokens are
    prepended to the noisy displacement tokens, and the main denoiser then runs
    with timestep conditioning only.
    """

    def __init__(self, *args, prefix_tokens: int = 4, num_heads: int = 8, dropout: float = 0.0, **kwargs):
        super().__init__(*args, num_heads=num_heads, dropout=dropout, **kwargs)
        self.prefix_tokens = int(prefix_tokens)
        self.prefix_query = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * self.prefix_tokens),
        )
        self.prefix_pos_embed = nn.Parameter(torch.zeros(1, self.prefix_tokens, self.hidden_dim))
        self.prefix_query_norm = nn.LayerNorm(self.hidden_dim)
        self.prefix_key_norm = nn.LayerNorm(self.hidden_dim)
        self.prefix_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.prefix_out_norm = nn.LayerNorm(self.hidden_dim)
        nn.init.trunc_normal_(self.prefix_pos_embed, std=0.02)

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, _, anchor_tokens = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        batch_size = h.shape[0]
        condition_seed = condition_tokens[:, 0, :]
        query_tokens = self.prefix_query(condition_seed).view(batch_size, self.prefix_tokens, self.hidden_dim)
        query_tokens = query_tokens + self.prefix_pos_embed
        prefix_delta, _ = self.prefix_cross_attn(
            self.prefix_query_norm(query_tokens),
            self.prefix_key_norm(anchor_tokens),
            self.prefix_key_norm(anchor_tokens),
            need_weights=False,
        )
        prefix_tokens = self.prefix_out_norm(query_tokens + prefix_delta)
        h = torch.cat([prefix_tokens, h], dim=1)
        time_cond = time_tokens[:, :1, :].expand(-1, h.shape[1], -1)
        for block in self.blocks:
            h = block(h, time_cond)
        h = h[:, self.prefix_tokens :, :]
        return self.out_proj(self.norm(h))


class BilinearOnceConditionDisplacementDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    One-shot global GO x control interaction.

    The control anchor is pooled once, fused with GO through a small bilinear-
    style MLP, and injected only into the input tokens. The denoiser blocks
    then run with timestep conditioning only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bilinear_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, _, anchor_tokens = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        condition_summary = condition_tokens[:, 0, :]
        pooled_anchor = anchor_tokens.mean(dim=1)
        fused = self.bilinear_fusion(
            torch.cat(
                [
                    condition_summary,
                    pooled_anchor,
                    condition_summary * pooled_anchor,
                    torch.abs(condition_summary - pooled_anchor),
                ],
                dim=-1,
            )
        )
        h = h + fused[:, None, :]
        for block in self.blocks:
            h = block(h, time_tokens)
        return self.out_proj(self.norm(h))


class TokenFusionControlAnchoredDisplacementDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    CatCrossDiT-style ablation: inject GO/control-anchor tokens into the
    denoising stream directly, in addition to the original AdaLN conditioning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h, time_tokens, condition_tokens, context_tokens, anchor_tokens = self._encode_condition_stream(
            u_t=u_t,
            timesteps=timesteps,
            condition_features=condition_features,
            context_features=context_features,
            control_anchor=control_anchor,
        )
        interaction_tokens = self.interaction_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )
        fusion_tokens = condition_tokens + context_tokens + anchor_tokens + interaction_tokens
        h = h + self.fusion_scale.to(dtype=h.dtype) * fusion_tokens
        cond = time_tokens + fusion_tokens
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h))


class CrossAnchorTransformerBlock(nn.Module):
    """Self-attention over noisy displacement tokens plus cross-attention to anchor memory."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, cond_tokens, memory_tokens):
        chunks = self.adaLN(cond_tokens).chunk(9, dim=-1)
        (
            shift_self,
            scale_self,
            gate_self,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = chunks
        self_in = self.modulate(self.norm_self(x), shift_self, scale_self)
        self_out, _ = self.self_attn(self_in, self_in, self_in, need_weights=False)
        x = x + gate_self * self.dropout(self_out)
        cross_in = self.modulate(self.norm_cross(x), shift_cross, scale_cross)
        cross_out, _ = self.cross_attn(cross_in, memory_tokens, memory_tokens, need_weights=False)
        x = x + gate_cross * self.dropout(cross_out)
        mlp_in = self.modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.dropout(self.mlp(mlp_in))
        return x


class CrossAnchorDisplacementDenoiser(nn.Module):
    """
    CrossDiT-style ablation: noisy displacement tokens self-attend, then
    cross-attend to a GO-modulated control-anchor memory.
    """

    def __init__(
        self,
        num_patches: int,
        latent_dim: int,
        condition_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = int(num_patches)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.in_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(latent_dim, hidden_dim)
        self.out_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(hidden_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_feature_dim),
            nn.Linear(condition_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_proj = nn.Identity() if latent_dim == hidden_dim else nn.Linear(latent_dim, hidden_dim)
        self.interaction_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        ff_dim = int(hidden_dim * mlp_ratio)
        self.blocks = nn.ModuleList(
            [CrossAnchorTransformerBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h = self.in_proj(u_t) + self.pos_embed
        time_tokens = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim).to(h.dtype))[:, None, :]
        condition_tokens = self.condition_encoder(condition_features.to(dtype=h.dtype))[:, None, :]
        context_tokens = self.context_encoder(context_features.to(dtype=h.dtype))[:, None, :]
        anchor_tokens = self.anchor_proj(control_anchor.to(dtype=h.dtype))
        condition_tokens = condition_tokens.expand(-1, h.shape[1], -1)
        context_tokens = context_tokens.expand(-1, h.shape[1], -1)
        time_tokens = time_tokens.expand(-1, h.shape[1], -1)
        interaction_tokens = self.interaction_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )
        cond = time_tokens + condition_tokens + context_tokens + interaction_tokens
        memory = anchor_tokens + condition_tokens + context_tokens + interaction_tokens + self.pos_embed
        for block in self.blocks:
            h = block(h, cond, memory)
        return self.out_proj(self.norm(h))


class GOControlPriorFeatureDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    Ablation where GO and the control-anchor token are first mapped through a
    small token-wise network. The network output is used only as a conditioning
    feature; the diffusion target remains the full displacement u.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_feature_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _prior_feature_tokens(self, condition_features, control_anchor, dtype):
        condition_tokens = self.condition_encoder(condition_features.to(dtype=dtype))[:, None, :]
        anchor_tokens = self.anchor_proj(control_anchor.to(dtype=dtype))
        condition_tokens = condition_tokens.expand(-1, anchor_tokens.shape[1], -1)
        return self.prior_feature_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h = self.in_proj(u_t) + self.pos_embed
        time_tokens = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim).to(h.dtype))[:, None, :]
        context_tokens = self.context_encoder(context_features.to(dtype=h.dtype))[:, None, :]
        prior_tokens = self._prior_feature_tokens(condition_features, control_anchor, h.dtype)
        time_tokens = time_tokens.expand(-1, h.shape[1], -1)
        context_tokens = context_tokens.expand(-1, h.shape[1], -1)
        cond = time_tokens + context_tokens + prior_tokens
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.norm(h))


class DirectAddConditionDenoiser(ControlAnchoredDisplacementDenoiser):
    """
    Conditioning ablation without AdaLN.

    GO and control-anchor features are fused into patch-wise condition tokens,
    then directly added to the noisy latent tokens before each plain
    Transformer block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ff_dim = int(self.hidden_dim * 4)
        # Preserve the requested mlp_ratio from the already-built AdaLN blocks.
        if len(self.blocks) > 0:
            first_mlp = self.blocks[0].mlp
            ff_dim = int(first_mlp[0].out_features)
        self.blocks = nn.ModuleList(
            [
                PlainTransformerBlock(
                    self.hidden_dim,
                    self.blocks[0].attn.num_heads if len(self.blocks) > 0 else 8,
                    ff_dim,
                    self.blocks[0].dropout.p if len(self.blocks) > 0 else 0.0,
                )
                for _ in range(len(self.blocks))
            ]
        )
        self.prior_feature_mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _prior_feature_tokens(self, condition_features, control_anchor, dtype):
        condition_tokens = self.condition_encoder(condition_features.to(dtype=dtype))[:, None, :]
        anchor_tokens = self.anchor_proj(control_anchor.to(dtype=dtype))
        condition_tokens = condition_tokens.expand(-1, anchor_tokens.shape[1], -1)
        return self.prior_feature_mlp(
            torch.cat([condition_tokens, anchor_tokens, condition_tokens * anchor_tokens], dim=-1)
        )

    def forward(self, u_t, timesteps, condition_features, context_features, control_anchor):
        h = self.in_proj(u_t) + self.pos_embed
        time_tokens = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim).to(h.dtype))[:, None, :]
        context_tokens = self.context_encoder(context_features.to(dtype=h.dtype))[:, None, :]
        prior_tokens = self._prior_feature_tokens(condition_features, control_anchor, h.dtype)
        time_tokens = time_tokens.expand(-1, h.shape[1], -1)
        context_tokens = context_tokens.expand(-1, h.shape[1], -1)
        cond = time_tokens + context_tokens + prior_tokens
        for block in self.blocks:
            h = block(h + cond)
        return self.out_proj(self.norm(h))


class DiffusionHelper:
    def __init__(self, steps: int, schedule: str, device):
        betas = make_beta_schedule(schedule, steps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.steps = int(steps)
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32, device=device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32, device=device
        )

    def q_sample(self, z_start, t, noise):
        shape = (z_start.shape[0],) + (1,) * (z_start.ndim - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(shape)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(shape)
        return sqrt_alpha * z_start + sqrt_om * noise
