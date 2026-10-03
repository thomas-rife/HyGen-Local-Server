"""
Conditional Latent Diffusion Model (cLDM) + ControlNet for Terrain Generation.

Architecture:
  Phase 1: HeightmapVAE    — compresses 256x256 heightmaps to 32x32 latents
  Phase 2: TerrainUNet     — text-conditioned denoiser in latent space
  Phase 3: TerrainControlNet — border-conditioned copy of encoder for outpainting

This file defines model initialization. Training loops are separate.

Author: ML Architect Roadmap
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimestepEmbedding(nn.Module):
    """Maps scalar diffusion timestep to a vector embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class AdaptiveGroupNorm(nn.Module):
    """GroupNorm with scale/shift modulated by timestep embedding."""

    def __init__(self, num_channels: int, emb_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)
        self.proj = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(emb)[:, :, None, None].chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual block with timestep conditioning via AdaptiveGroupNorm."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = AdaptiveGroupNorm(in_ch, emb_dim, num_groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(out_ch, emb_dim, num_groups)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x, emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, emb))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention for text conditioning."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(context_dim, channels)
        self.v = nn.Linear(context_dim, channels)
        self.out = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Reshape spatial to sequence
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]

        q = self.q(h)
        k = self.k(context)
        v = self.v(context)

        # Multi-head reshape
        def split_heads(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, H * W, C)
        attn_out = self.out(attn_out)

        return residual + attn_out.permute(0, 2, 1).reshape(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: HEIGHTMAP VAE
# ══════════════════════════════════════════════════════════════════════════════

class HeightmapVAE(nn.Module):
    """
    Compresses 1-channel 256x256 heightmaps to 4-channel 32x32 latents.
    Downsampling factor: 8 (three 2x downsamples).

    Train this FIRST with reconstruction loss + KL divergence.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.latent_channels = latent_channels
        channels = [base_channels * m for m in channel_mults]

        # ── Encoder ──
        enc_layers = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        for i in range(len(channels)):
            in_c = channels[i]
            out_c = channels[min(i + 1, len(channels) - 1)]
            enc_layers += [
                nn.GroupNorm(32, in_c),
                nn.SiLU(),
                nn.Conv2d(in_c, in_c, 3, padding=1),
                nn.GroupNorm(32, in_c),
                nn.SiLU(),
                nn.Conv2d(in_c, out_c, 3, padding=1),
            ]
            if i < len(channels) - 1:
                enc_layers.append(nn.Conv2d(out_c, out_c, 3, stride=2, padding=1))

        enc_layers += [
            nn.GroupNorm(32, channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], latent_channels * 2, 1),  # mean + logvar
        ]
        self.encoder = nn.Sequential(*enc_layers)

        # ── Decoder ──
        dec_layers = [nn.Conv2d(latent_channels, channels[-1], 1)]
        for i in range(len(channels) - 1, -1, -1):
            in_c = channels[i]
            out_c = channels[max(i - 1, 0)]
            dec_layers += [
                nn.GroupNorm(32, in_c),
                nn.SiLU(),
                nn.Conv2d(in_c, in_c, 3, padding=1),
                nn.GroupNorm(32, in_c),
                nn.SiLU(),
                nn.Conv2d(in_c, out_c, 3, padding=1),
            ]
            if i > 0:
                dec_layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                               nn.Conv2d(out_c, out_c, 3, padding=1)]

        dec_layers += [
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, padding=1),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: CONDITIONAL U-NET (text-conditioned denoiser in latent space)
# ══════════════════════════════════════════════════════════════════════════════

class TerrainUNet(nn.Module):
    """
    U-Net denoiser operating on 32x32 latent maps.

    Conditioning:
      - Timestep: sinusoidal embedding → AdaptiveGroupNorm
      - Text: cross-attention at 16x16 and 8x8 resolution
      - Biome: learned embedding summed into input (optional auxiliary)

    Architecture: 3 resolution levels (32→16→8) with skip connections.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        text_context_dim: int = 512,      # T5-small = 512, CLIP = 768
        num_biome_ids: int = 256,         # max biome vocabulary size
        biome_emb_dim: int = 0,           # 0 = no biome input to U-Net
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.attention_resolutions = attention_resolutions

        channels = [base_channels * m for m in channel_mults]
        emb_dim = base_channels * 4

        # ── Timestep embedding ──
        self.time_embed = SinusoidalTimestepEmbedding(emb_dim)

        # ── Optional biome embedding (concatenated as extra input channels) ──
        self.biome_emb_dim = biome_emb_dim
        if biome_emb_dim > 0:
            self.biome_embedding = nn.Embedding(num_biome_ids, biome_emb_dim)
            # A small encoder to downsample biome maps from 256x256 to 32x32
            self.biome_encoder = nn.Sequential(
                nn.Conv2d(biome_emb_dim, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            )
            in_ch = latent_channels + 32
        else:
            in_ch = latent_channels

        # ── Input projection ──
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ── Encoder (downsampling path) ──
        self.enc_blocks = nn.ModuleList()
        self.enc_downsamples = nn.ModuleList()
        current_res = 32  # latent spatial resolution

        for level, ch in enumerate(channels):
            prev_ch = channels[max(level - 1, 0)] if level > 0 else channels[0]
            level_blocks = nn.ModuleList()

            for blk_idx in range(num_res_blocks):
                block_in = prev_ch if blk_idx == 0 and level > 0 else ch
                level_blocks.append(ResBlock(block_in, ch, emb_dim))
                if current_res in attention_resolutions:
                    level_blocks.append(CrossAttentionBlock(ch, text_context_dim))

            self.enc_blocks.append(level_blocks)

            if level < len(channels) - 1:
                self.enc_downsamples.append(Downsample(ch))
                current_res //= 2
            else:
                self.enc_downsamples.append(nn.Identity())

        # ── Bottleneck ──
        self.mid_block1 = ResBlock(channels[-1], channels[-1], emb_dim)
        self.mid_attn   = CrossAttentionBlock(channels[-1], text_context_dim)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], emb_dim)

        # ── Decoder (upsampling path) ──
        self.dec_blocks = nn.ModuleList()
        self.dec_upsamples = nn.ModuleList()

        for level in range(len(channels) - 1, -1, -1):
            ch = channels[level]
            # Skip connection doubles the input channels
            skip_ch = ch  # from encoder
            level_blocks = nn.ModuleList()

            for blk_idx in range(num_res_blocks):
                block_in = (ch + skip_ch) if blk_idx == 0 else ch
                level_blocks.append(ResBlock(block_in, ch, emb_dim))
                if current_res in attention_resolutions:
                    level_blocks.append(CrossAttentionBlock(ch, text_context_dim))

            self.dec_blocks.append(level_blocks)

            if level > 0:
                self.dec_upsamples.append(Upsample(ch))
                prev_ch = channels[level - 1]
                self.dec_upsamples.append(nn.Conv2d(ch, prev_ch, 1))
                current_res *= 2
            else:
                self.dec_upsamples.append(nn.Identity())
                self.dec_upsamples.append(nn.Identity())

        # ── Output projection ──
        self.out_norm = nn.GroupNorm(32, channels[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(channels[0], latent_channels, 3, padding=1)

    def forward(
        self,
        z_noisy: torch.Tensor,          # [B, latent_ch, 32, 32]
        t: torch.Tensor,                # [B] integer timesteps
        text_emb: torch.Tensor,         # [B, seq_len, context_dim]
        biome_ids: Optional[torch.Tensor] = None,  # [B, 1, 256, 256] int64
        controlnet_residuals: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict noise ε given noisy latent, timestep, and conditioning."""

        emb = self.time_embed(t)

        # Optional biome conditioning
        if self.biome_emb_dim > 0 and biome_ids is not None:
            B, _, H, W = biome_ids.shape
            biome_flat = biome_ids.squeeze(1)  # [B, H, W]
            biome_vec = self.biome_embedding(biome_flat)  # [B, H, W, emb_dim]
            biome_vec = biome_vec.permute(0, 3, 1, 2)     # [B, emb_dim, H, W]
            biome_feat = self.biome_encoder(biome_vec)     # [B, 32, 32, 32]
            z_noisy = torch.cat([z_noisy, biome_feat], dim=1)

        h = self.input_conv(z_noisy)

        # ── Encoder ──
        skips = []
        for level, (blocks, down) in enumerate(zip(self.enc_blocks, self.enc_downsamples)):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                elif isinstance(block, CrossAttentionBlock):
                    h = block(h, text_emb)
            skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)

        # ── Bottleneck ──
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h, text_emb)
        h = self.mid_block2(h, emb)

        # ── Decoder ──
        # ControlNet residuals order: [bottleneck, level_N-1, ..., level_0]
        # First residual goes to bottleneck h, rest go to skip connections
        ctrl_idx = 0
        if controlnet_residuals is not None and ctrl_idx < len(controlnet_residuals):
            h = h + controlnet_residuals[ctrl_idx]
            ctrl_idx += 1

        for level_idx, blocks in enumerate(self.dec_blocks):
            # Pop the matching skip connection
            skip = skips.pop()

            # Add ControlNet residual to the skip connection (before concat)
            if controlnet_residuals is not None and ctrl_idx < len(controlnet_residuals):
                skip = skip + controlnet_residuals[ctrl_idx]
                ctrl_idx += 1

            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                elif isinstance(block, CrossAttentionBlock):
                    h = block(h, text_emb)

            # Upsample
            up_pair = self.dec_upsamples[level_idx * 2 : level_idx * 2 + 2]
            for up in up_pair:
                if not isinstance(up, nn.Identity):
                    h = up(h)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: CONTROLNET FOR BORDER-CONDITIONED OUTPAINTING
# ══════════════════════════════════════════════════════════════════════════════

class ZeroConv(nn.Module):
    """1x1 conv initialized to zero — the ControlNet injection mechanism."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BorderEncoder(nn.Module):
    """
    Encodes the 256x256 border_context + border_mask into 64x64 feature maps
    that match the latent spatial resolution (4x downsample).
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        # Input: 2 channels (border_context + border_mask)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),   # 128x128
            nn.SiLU(),
            nn.Conv2d(32, out_channels, 3, stride=2, padding=1),  # 64x64
            nn.SiLU(),
        )

    def forward(self, border_context: torch.Tensor, border_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([border_context, border_mask], dim=1)  # [B, 2, 256, 256]
        return self.net(x)  # [B, out_ch, 64, 64]


class TerrainControlNet(nn.Module):
    """
    ControlNet for border-conditioned outpainting.

    Clones the encoder path of the frozen TerrainUNet and adds:
      1. A BorderEncoder to project raw border context into latent space
      2. Zero-initialized convolutions at each skip connection output

    At inference:
      - Encode border strips → border features
      - Sum border features with noisy latent as ControlNet input
      - Run through cloned encoder → produce residuals
      - Inject residuals into the frozen U-Net's decoder via addition

    Training: Freeze the base U-Net, train only the ControlNet + BorderEncoder.
    """

    def __init__(
        self,
        unet: TerrainUNet,
        border_encoder_channels: int = 64,
    ):
        super().__init__()

        # ── Border encoder (raw pixels → latent-resolution features) ──
        self.border_encoder = BorderEncoder(out_channels=border_encoder_channels)

        # Project border features to match latent input channels
        self.border_proj = nn.Conv2d(
            border_encoder_channels, unet.base_channels, 1
        )

        # ── Clone the U-Net's encoder blocks ──
        import copy
        self.input_conv = copy.deepcopy(unet.input_conv)
        self.enc_blocks = copy.deepcopy(unet.enc_blocks)
        self.enc_downsamples = copy.deepcopy(unet.enc_downsamples)
        self.mid_block1 = copy.deepcopy(unet.mid_block1)
        self.mid_attn   = copy.deepcopy(unet.mid_attn)
        self.mid_block2 = copy.deepcopy(unet.mid_block2)

        # ── Zero convolutions: one per encoder level + one for bottleneck ──
        channels = [unet.base_channels * m for m in (1, 2, 4)]  # must match UNet channel_mults
        self.zero_convs = nn.ModuleList()
        for ch in channels:
            self.zero_convs.append(ZeroConv(ch, ch))
        self.zero_convs.append(ZeroConv(channels[-1], channels[-1]))  # bottleneck

        # Copy timestep embedding
        self.time_embed = copy.deepcopy(unet.time_embed)

    def forward(
        self,
        z_noisy: torch.Tensor,           # [B, latent_ch, 32, 32]
        t: torch.Tensor,                 # [B]
        text_emb: torch.Tensor,          # [B, seq_len, context_dim]
        border_context: torch.Tensor,    # [B, 1, 256, 256]
        border_mask: torch.Tensor,       # [B, 1, 256, 256]
    ) -> List[torch.Tensor]:
        """Returns list of residual tensors to inject into the U-Net decoder."""

        emb = self.time_embed(t)

        # Encode border context → latent-resolution features
        border_feat = self.border_encoder(border_context, border_mask)  # [B, 64, 32, 32]
        border_feat = self.border_proj(border_feat)  # [B, base_ch, 32, 32]

        h = self.input_conv(z_noisy)
        h = h + border_feat  # Inject border conditioning

        residuals = []
        for level, (blocks, down, zero_conv) in enumerate(
            zip(self.enc_blocks, self.enc_downsamples, self.zero_convs[:-1])
        ):
            for block in blocks:
                if isinstance(block, ResBlock):
                    h = block(h, emb)
                elif isinstance(block, CrossAttentionBlock):
                    h = block(h, text_emb)
            residuals.append(zero_conv(h))
            if not isinstance(down, nn.Identity):
                h = down(h)

        # Bottleneck
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h, text_emb)
        h = self.mid_block2(h, emb)
        residuals.append(self.zero_convs[-1](h))

        # Reverse so residuals[0] corresponds to the first decoder level
        return residuals[::-1]


# ══════════════════════════════════════════════════════════════════════════════
# DIFFUSION SCHEDULE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class CosineNoiseSchedule:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).
    Provides alpha_bar lookup for training and sampling.
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        self.T = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar = alpha_bar.float()

        # Precompute betas and clip
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.betas = torch.clamp(betas.float(), max=0.999)
        self.alphas = 1.0 - self.betas

    def q_sample(
        self, z0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean latent z0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(z0)
        ab = self.alpha_bar[t.cpu()].to(z0.device)
        while ab.dim() < z0.dim():
            ab = ab.unsqueeze(-1)
        z_noisy = ab.sqrt() * z0 + (1 - ab).sqrt() * noise
        return z_noisy, noise


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY — initialize the full system
# ══════════════════════════════════════════════════════════════════════════════

def build_terrain_cldm(
    latent_channels: int = 4,
    unet_base_channels: int = 128,
    text_context_dim: int = 512,
    device: str = "cuda",
) -> dict:
    """
    Instantiate all model components for the terrain generation pipeline.

    Returns a dict with:
      vae, unet, controlnet, schedule, and parameter counts.
    """
    vae = HeightmapVAE(
        in_channels=1,
        latent_channels=latent_channels,
        base_channels=64,
        channel_mults=(1, 2, 4),  # 3 levels → 2 downsamples → 256/4 = 64
    ).to(device)

    unet = TerrainUNet(
        latent_channels=latent_channels,
        base_channels=unet_base_channels,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16),
        text_context_dim=text_context_dim,
    ).to(device)

    controlnet = TerrainControlNet(
        unet=unet,
        border_encoder_channels=64,
    ).to(device)

    schedule = CosineNoiseSchedule(num_timesteps=1000)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    info = {
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "schedule": schedule,
        "param_counts": {
            "vae":        f"{count_params(vae) / 1e6:.1f}M",
            "unet":       f"{count_params(unet) / 1e6:.1f}M",
            "controlnet": f"{count_params(controlnet) / 1e6:.1f}M",
        },
    }
    return info


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    models = build_terrain_cldm(device=device)

    print("Parameter counts:")
    for name, count in models["param_counts"].items():
        print(f"  {name:15s} {count}")

    # ── Test VAE ──
    print("\n--- VAE Forward Pass ---")
    x = torch.randn(2, 1, 256, 256, device=device)
    recon, mean, logvar = models["vae"](x)
    print(f"  Input:  {x.shape}")
    print(f"  Latent: {mean.shape}")
    print(f"  Recon:  {recon.shape}")
    print(f"  KL:     {HeightmapVAE.kl_loss(mean, logvar):.4f}")

    # ── Test U-Net ──
    print("\n--- U-Net Forward Pass ---")
    z = torch.randn(2, 4, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    text = torch.randn(2, 16, 512, device=device)  # fake text embeddings
    noise_pred = models["unet"](z, t, text)
    print(f"  z_noisy:    {z.shape}")
    print(f"  noise_pred: {noise_pred.shape}")

    # ── Test ControlNet ──
    print("\n--- ControlNet Forward Pass ---")
    border_ctx  = torch.randn(2, 1, 256, 256, device=device)
    border_mask = torch.zeros(2, 1, 256, 256, device=device)
    border_mask[:, :, :16, :] = 1.0  # top border active
    residuals = models["controlnet"](z, t, text, border_ctx, border_mask)
    print(f"  Residuals: {len(residuals)} tensors")
    for i, r in enumerate(residuals):
        print(f"    [{i}] {r.shape}")

    # ── Test with ControlNet injection into U-Net ──
    print("\n--- Combined Forward Pass (U-Net + ControlNet) ---")
    noise_pred_conditioned = models["unet"](z, t, text, controlnet_residuals=residuals)
    print(f"  Conditioned noise_pred: {noise_pred_conditioned.shape}")

    print("\n✓ All components initialized and forward passes verified.")
