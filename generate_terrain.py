"""
generate_terrain.py — Phase 4: Standalone inference for the terrain cLDM + ControlNet.

Generates a GxG grid of 256x256 terrain chunks around a flat center "arena".
The center chunk is procedural (flat at Y=63). Surrounding chunks are generated
AI-side, one at a time, using a Chebyshev-ring spiral order so each chunk is
conditioned on every already-generated neighbor it has.

Pipeline per AI chunk:
    1. Build border_context + border_mask from already-generated neighbors.
    2. DDIM-sample a 64x64 latent with ControlNet injection + CFG.
    3. VAE-decode to a normalized [-1, 1] 256x256 heightmap.
    4. Store; later chunks use its edges as border context.

Final step:
    - Stitch all chunks into a (256*G) x (256*G) heightmap.
    - Denormalize to raw Y-levels.
    - Render with matplotlib + chunk grid overlay + seam L1 diagnostic.
    - Optionally save .npy.

Usage (single prompt, whole world):
    python generate_terrain.py \
        --prompt "alpine mountains" \
        --grid_size 3 \
        --vae_ckpt vae_ema_final.pt \
        --unet_ckpt unet_ema_final.pt \
        --controlnet_ckpt controlnet_ema_final.pt \
        --output terrain.png

Usage (directional prompts):
    python generate_terrain.py \
        --prompt_north "steep mountain peaks" \
        --prompt_south "deep ocean with coastline" \
        --prompt_east  "dense forest" \
        --prompt_west  "desert canyon" \
        --grid_size 5 \
        --vae_ckpt vae_ema_final.pt \
        --unet_ckpt unet_ema_final.pt \
        --controlnet_ckpt controlnet_ema_final.pt \
        --output terrain.png
"""

import os
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast

# Third-party — we only import matplotlib at visualization time so the main
# pipeline still runs on headless systems without it.
from terrain_cldm import (
    HeightmapVAE,
    TerrainUNet,
    TerrainControlNet,
    CosineNoiseSchedule,
)
from terrain_dataloader import (
    normalize_height,
    denormalize_height,
    HEIGHT_GLOBAL_MIN,
    HEIGHT_GLOBAL_MAX,
)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE = 256
BORDER_WIDTH = 16
LATENT_SIZE = 64
LATENT_CHANNELS = 4
CENTER_Y_LEVEL = 63  # sea level — the flat arena


# ══════════════════════════════════════════════════════════════════════════════
# T5 TEXT ENCODER (mirrors train_controlnet.py / train_unet.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class T5TextEncoder:
    """Frozen T5-small encoder for caption → embedding [1, seq_len, 512]."""

    def __init__(self, device: str = "cuda", max_length: int = 128):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = device
        self.max_length = max_length

        print("Loading T5-small text encoder...")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5EncoderModel.from_pretrained("t5-small").to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        assert self.model.config.d_model == 512, (
            f"T5-small d_model={self.model.config.d_model}, expected 512."
        )

        self._null_embedding = self._encode_single("")
        print(f"  T5-small loaded. Output dim: {self.model.config.d_model}")

    @torch.inference_mode()
    def _encode_single(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids.to(self.device)
        return self.model(input_ids=tokens).last_hidden_state

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        """Encode a single caption → [1, seq_len, 512]."""
        return self._encode_single(text)

    @property
    def null_embedding(self) -> torch.Tensor:
        return self._null_embedding


# ══════════════════════════════════════════════════════════════════════════════
# DDIM SAMPLING (single-chunk, batch=1, ControlNet-guided)
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def ddim_sample_chunk(
    unet: TerrainUNet,
    controlnet: TerrainControlNet,
    schedule: CosineNoiseSchedule,
    text_emb: torch.Tensor,           # [1, seq_len, 512]
    null_text_emb: torch.Tensor,      # [1, seq_len, 512]
    border_context: torch.Tensor,     # [1, 1, 256, 256]
    border_mask: torch.Tensor,        # [1, 1, 256, 256]
    device: str,
    num_steps: int = 50,
    cfg_scale: float = 5.0,
) -> torch.Tensor:
    """
    DDIM loop evaluating ControlNet + U-Net at every step. CFG uses the same
    border context for both conditional and unconditional passes — only the
    text embedding changes. Returns a final latent [1, 4, 64, 64].

    Device fix: schedule.alpha_bar lives on CPU. We do alpha_bar[t.cpu()].to(device)
    for any CUDA timestep and alpha_bar[t_prev] directly (already CPU) for the
    linspace-derived index.
    """
    unet.eval()
    controlnet.eval()

    step_indices = torch.linspace(schedule.T - 1, 0, num_steps, dtype=torch.long)

    z = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device)

    amp_device = device.split(":")[0]
    use_amp = device.startswith("cuda")

    for i, t_val in enumerate(step_indices):
        t = torch.full((1,), t_val.item(), device=device, dtype=torch.long)

        # Conditional pass
        if use_amp:
            with autocast(device_type=amp_device, dtype=torch.float16):
                residuals_c = controlnet(z, t, text_emb, border_context, border_mask)
                eps_cond = unet(z, t, text_emb, controlnet_residuals=residuals_c)
        else:
            residuals_c = controlnet(z, t, text_emb, border_context, border_mask)
            eps_cond = unet(z, t, text_emb, controlnet_residuals=residuals_c)

        # Unconditional pass (CFG) — same borders, null text
        if cfg_scale > 1.0:
            if use_amp:
                with autocast(device_type=amp_device, dtype=torch.float16):
                    residuals_u = controlnet(z, t, null_text_emb, border_context, border_mask)
                    eps_uncond = unet(z, t, null_text_emb, controlnet_residuals=residuals_u)
            else:
                residuals_u = controlnet(z, t, null_text_emb, border_context, border_mask)
                eps_uncond = unet(z, t, null_text_emb, controlnet_residuals=residuals_u)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        eps = eps.float()

        # DDIM update — device fix on alpha_bar indexing
        ab_t = schedule.alpha_bar[t.cpu()].to(device)
        while ab_t.dim() < z.dim():
            ab_t = ab_t.unsqueeze(-1)

        if i < len(step_indices) - 1:
            t_prev = step_indices[i + 1]  # already on CPU — no .cpu() needed
            ab_prev = schedule.alpha_bar[t_prev].to(device)
            while ab_prev.dim() < z.dim():
                ab_prev = ab_prev.unsqueeze(-1)
        else:
            ab_prev = torch.ones_like(ab_t)

        x0_pred = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0_pred = torch.clamp(x0_pred, -3, 3)
        dir_zt = (1 - ab_prev).sqrt() * eps
        z = ab_prev.sqrt() * x0_pred + dir_zt

    return z


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_models(
    vae_ckpt: str,
    unet_ckpt: str,
    controlnet_ckpt: str,
    device: str,
) -> Tuple[HeightmapVAE, TerrainUNet, TerrainControlNet]:
    """Instantiate all three models and load their EMA weights."""

    print(f"\nLoading VAE from:        {vae_ckpt}")
    vae = HeightmapVAE(
        in_channels=1,
        latent_channels=4,
        base_channels=64,
        channel_mults=(1, 2, 4),
    ).to(device)
    vae_state = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state["model"] if "model" in vae_state else vae_state)
    vae.eval()
    vae.requires_grad_(False)

    # Sanity-check latent shape
    with torch.inference_mode():
        probe = torch.randn(1, 1, CHUNK_SIZE, CHUNK_SIZE, device=device)
        mean, _ = vae.encode(probe)
        assert mean.shape == (1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), (
            f"VAE latent shape mismatch: got {mean.shape}, "
            f"expected (1, {LATENT_CHANNELS}, {LATENT_SIZE}, {LATENT_SIZE})"
        )
        del probe, mean
    print(f"  VAE latent shape verified: [B, {LATENT_CHANNELS}, {LATENT_SIZE}, {LATENT_SIZE}]")

    print(f"\nLoading U-Net from:      {unet_ckpt}")
    unet = TerrainUNet(
        latent_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16),  # stale label but must match training
        text_context_dim=512,
    ).to(device)
    unet_state = torch.load(unet_ckpt, map_location=device, weights_only=False)
    unet.load_state_dict(unet_state["model"] if "model" in unet_state else unet_state)
    unet.eval()
    unet.requires_grad_(False)

    print(f"\nLoading ControlNet from: {controlnet_ckpt}")
    # Build ControlNet from the U-Net — this deep-copies the encoder path.
    # Values in the deepcopied params don't matter here because we immediately
    # overwrite with the trained ControlNet state dict.
    controlnet = TerrainControlNet(
        unet=unet,
        border_encoder_channels=64,
    ).to(device)
    cn_state = torch.load(controlnet_ckpt, map_location=device, weights_only=False)
    controlnet.load_state_dict(cn_state["model"] if "model" in cn_state else cn_state)
    controlnet.eval()
    controlnet.requires_grad_(False)

    # Log param counts
    def mp(m): return sum(p.numel() for p in m.parameters()) / 1e6
    print(f"\nParam counts — vae: {mp(vae):.1f}M  unet: {mp(unet):.1f}M  "
          f"controlnet: {mp(controlnet):.1f}M")

    return vae, unet, controlnet


# ══════════════════════════════════════════════════════════════════════════════
# BORDER CONTEXT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_border_context(
    chunk_rc: Tuple[int, int],
    chunks: Dict[Tuple[int, int], torch.Tensor],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    For a chunk at (row, col), look at the four orthogonal neighbors. For each
    neighbor that is already generated, extract a 16-pixel strip from the edge
    of the neighbor that FACES this chunk and place it at the corresponding
    edge of our border_context tensor.

    Convention (from the dataloader — border strip is on the INSIDE of the
    chunk that will be generated, at its edge toward the neighbor):

      neighbor ABOVE  (row-1, col) → neighbor's BOTTOM 16 rows → our TOP    edge
      neighbor BELOW  (row+1, col) → neighbor's TOP    16 rows → our BOTTOM edge
      neighbor LEFT   (row, col-1) → neighbor's RIGHT  16 cols → our LEFT   edge
      neighbor RIGHT  (row, col+1) → neighbor's LEFT   16 cols → our RIGHT  edge

    Chunks in `chunks` are stored as [1, 1, 256, 256] tensors in normalized
    [-1, 1] range (either VAE-decoded output or the flat normalized center).

    Returns (border_context, border_mask, active_edges_list).
    """
    r, c = chunk_rc
    bw = BORDER_WIDTH
    cs = CHUNK_SIZE

    ctx  = torch.zeros(1, 1, cs, cs, device=device)
    mask = torch.zeros(1, 1, cs, cs, device=device)
    active: List[str] = []

    above = chunks.get((r - 1, c))
    if above is not None:
        strip = above[:, :, -bw:, :].clamp(-1.0, 1.0)        # bottom of neighbor
        ctx[:, :, :bw, :] = strip
        mask[:, :, :bw, :] = 1.0
        active.append("top")

    below = chunks.get((r + 1, c))
    if below is not None:
        strip = below[:, :, :bw, :].clamp(-1.0, 1.0)         # top of neighbor
        ctx[:, :, -bw:, :] = strip
        mask[:, :, -bw:, :] = 1.0
        active.append("bottom")

    left = chunks.get((r, c - 1))
    if left is not None:
        strip = left[:, :, :, -bw:].clamp(-1.0, 1.0)         # right of neighbor
        ctx[:, :, :, :bw] = strip
        mask[:, :, :, :bw] = 1.0
        active.append("left")

    right = chunks.get((r, c + 1))
    if right is not None:
        strip = right[:, :, :, :bw].clamp(-1.0, 1.0)         # left of neighbor
        ctx[:, :, :, -bw:] = strip
        mask[:, :, :, -bw:] = 1.0
        active.append("right")

    return ctx, mask, active


# ══════════════════════════════════════════════════════════════════════════════
# SPIRAL ORDER
# ══════════════════════════════════════════════════════════════════════════════

def spiral_order(grid_size: int) -> List[Tuple[int, int]]:
    """
    Return (row, col) coordinates in Chebyshev-ring order around the center,
    walking clockwise within each ring starting from SOUTH of center.

    For grid_size=3 (center at (1,1)) this gives:
        [(2,1), (2,2), (1,2), (0,2), (0,1), (0,0), (1,0), (2,0)]
    which matches the user's specified order (south → SE → east → NE → north
    → NW → west → SW).

    Center cell is EXCLUDED (it's not AI-generated).
    """
    cr = cc = grid_size // 2
    max_ring = max(cr, cc, grid_size - 1 - cr, grid_size - 1 - cc)

    order: List[Tuple[int, int]] = []
    for ring in range(1, max_ring + 1):
        # Start at directly south of center, at this ring's radius.
        # Walk: south → east along bottom, east → north along right,
        #       north → west along top, west → south along left.
        # Only include cells actually inside the grid.
        ring_cells: List[Tuple[int, int]] = []

        # South edge of the ring: row = cr + ring, col varies from cc to cc+ring (right half)
        # then we continue clockwise. Actually the cleanest description:
        # Start at (cr + ring, cc). Move right to (cr + ring, cc + ring).
        # Then up to (cr - ring, cc + ring).
        # Then left to (cr - ring, cc - ring).
        # Then down to (cr + ring, cc - ring).
        # Then right back to (cr + ring, cc - 1) (one short of start).

        r, c = cr + ring, cc
        ring_cells.append((r, c))

        # → right
        while c < cc + ring:
            c += 1
            ring_cells.append((r, c))
        # → up
        while r > cr - ring:
            r -= 1
            ring_cells.append((r, c))
        # → left
        while c > cc - ring:
            c -= 1
            ring_cells.append((r, c))
        # → down
        while r < cr + ring:
            r += 1
            ring_cells.append((r, c))
        # → right (close the ring, stopping one before start)
        while c < cc - 1:
            c += 1
            ring_cells.append((r, c))
        # Add the last step back toward start
        while c < cc:
            c += 1
            if (r, c) == (cr + ring, cc):
                break
            ring_cells.append((r, c))

        # Keep only in-bounds, non-center cells, preserving order, deduped.
        seen = set()
        for rc in ring_cells:
            if rc == (cr, cc):
                continue
            if not (0 <= rc[0] < grid_size and 0 <= rc[1] < grid_size):
                continue
            if rc in seen:
                continue
            seen.add(rc)
            order.append(rc)

    return order


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTIONAL PROMPT SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def pick_prompt_for_chunk(
    rc: Tuple[int, int],
    grid_size: int,
    single_prompt: Optional[str],
    directional: Dict[str, Optional[str]],
) -> str:
    """
    For single-prompt mode, every chunk gets the same prompt.

    For directional mode, pick N/S/E/W based on which cardinal direction the
    chunk sits from center. Corner chunks (|dx| == |dy|) default to N/S.
    Missing directions fall back to a reasonable default string.
    """
    if single_prompt is not None:
        return single_prompt

    cr = cc = grid_size // 2
    r, c = rc
    dy = r - cr  # +ve = south
    dx = c - cc  # +ve = east

    # Choose cardinal by dominant axis. Ties go to N/S.
    if abs(dy) >= abs(dx):
        direction = "south" if dy > 0 else "north"
    else:
        direction = "east" if dx > 0 else "west"

    fallback = "varied natural terrain"
    return directional.get(direction) or fallback


# ══════════════════════════════════════════════════════════════════════════════
# CENTER CHUNK (flat arena, not AI-generated)
# ══════════════════════════════════════════════════════════════════════════════

def make_center_chunk(device: str) -> torch.Tensor:
    """Flat chunk at Y = CENTER_Y_LEVEL, normalized to [-1, 1]."""
    # normalize_height accepts numpy arrays; compute the scalar then broadcast.
    flat_y = np.full((CHUNK_SIZE, CHUNK_SIZE), CENTER_Y_LEVEL, dtype=np.float32)
    flat_norm = normalize_height(flat_y)
    t = torch.from_numpy(flat_norm).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,256,256]
    return t


# ══════════════════════════════════════════════════════════════════════════════
# FULL GRID GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_grid(
    grid_size: int,
    vae: HeightmapVAE,
    unet: TerrainUNet,
    controlnet: TerrainControlNet,
    schedule: CosineNoiseSchedule,
    text_encoder: T5TextEncoder,
    device: str,
    cfg_scale: float,
    num_steps: int,
    single_prompt: Optional[str],
    directional: Dict[str, Optional[str]],
    seed: Optional[int],
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], List[str]]:
    """
    Generate all chunks. Returns:
        chunks_norm: {(row, col): [1, 1, 256, 256] normalized tensor}
        prompts_used: list of prompts in generation order for reporting
    """
    assert grid_size % 2 == 1, "grid_size must be odd so the center is well-defined"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    cr = cc = grid_size // 2
    chunks: Dict[Tuple[int, int], torch.Tensor] = {}

    # Center — procedural
    chunks[(cr, cc)] = make_center_chunk(device)
    print(f"\n  Center chunk at ({cr},{cc}): flat at Y={CENTER_Y_LEVEL}")

    order = spiral_order(grid_size)
    total = len(order)
    prompts_used: List[str] = []

    amp_device = device.split(":")[0]
    use_amp = device.startswith("cuda")

    for i, rc in enumerate(order, start=1):
        prompt = pick_prompt_for_chunk(rc, grid_size, single_prompt, directional)
        prompts_used.append(prompt)

        text_emb = text_encoder.encode(prompt)
        null_emb = text_encoder.null_embedding

        border_ctx, border_mask, active_edges = build_border_context(rc, chunks, device)

        print(f"  [{i:2d}/{total}] ({rc[0]},{rc[1]})  "
              f"borders: {active_edges or ['none']}  "
              f"prompt: \"{prompt[:48]}\"")

        z_final = ddim_sample_chunk(
            unet=unet,
            controlnet=controlnet,
            schedule=schedule,
            text_emb=text_emb,
            null_text_emb=null_emb,
            border_context=border_ctx,
            border_mask=border_mask,
            device=device,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )

        if use_amp:
            with autocast(device_type=amp_device, dtype=torch.float16):
                hm = vae.decode(z_final).float()
        else:
            hm = vae.decode(z_final)

        hm = hm.clamp(-1.0, 1.0)
        chunks[rc] = hm

        # Free intermediates
        del z_final
        if use_amp:
            torch.cuda.empty_cache()

    return chunks, prompts_used


# ══════════════════════════════════════════════════════════════════════════════
# STITCHING + SEAM DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def stitch_chunks(
    chunks: Dict[Tuple[int, int], torch.Tensor],
    grid_size: int,
) -> np.ndarray:
    """Returns [grid_size*256, grid_size*256] normalized heightmap (numpy)."""
    H = grid_size * CHUNK_SIZE
    out = np.zeros((H, H), dtype=np.float32)
    for (r, c), hm in chunks.items():
        patch = hm.squeeze(0).squeeze(0).detach().cpu().numpy()  # [256, 256]
        out[r * CHUNK_SIZE : (r + 1) * CHUNK_SIZE,
            c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE] = patch
    return out


def compute_seam_diagnostics(stitched_raw: np.ndarray, grid_size: int) -> Dict[str, float]:
    """
    For every adjacent pair of chunk boundaries in raw Y-levels, compute the
    mean absolute difference between the last row/col of one chunk and the
    first row/col of the neighbor.
    """
    diffs_h: List[float] = []  # horizontal seams (between vertically-adjacent chunks)
    diffs_v: List[float] = []  # vertical seams (between horizontally-adjacent chunks)

    for r in range(grid_size):
        for c in range(grid_size):
            # right seam
            if c + 1 < grid_size:
                col_right = c * CHUNK_SIZE + CHUNK_SIZE - 1
                a = stitched_raw[r * CHUNK_SIZE : (r + 1) * CHUNK_SIZE, col_right]
                b = stitched_raw[r * CHUNK_SIZE : (r + 1) * CHUNK_SIZE, col_right + 1]
                diffs_v.append(float(np.mean(np.abs(a - b))))
            # bottom seam
            if r + 1 < grid_size:
                row_bot = r * CHUNK_SIZE + CHUNK_SIZE - 1
                a = stitched_raw[row_bot,      c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
                b = stitched_raw[row_bot + 1,  c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
                diffs_h.append(float(np.mean(np.abs(a - b))))

    all_diffs = diffs_h + diffs_v
    return {
        "num_seams":     len(all_diffs),
        "mean_l1":       float(np.mean(all_diffs)) if all_diffs else 0.0,
        "max_l1":        float(np.max(all_diffs)) if all_diffs else 0.0,
        "median_l1":     float(np.median(all_diffs)) if all_diffs else 0.0,
        "horiz_mean_l1": float(np.mean(diffs_h)) if diffs_h else 0.0,
        "vert_mean_l1":  float(np.mean(diffs_v)) if diffs_v else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def render_png(
    stitched_raw: np.ndarray,
    grid_size: int,
    output_path: str,
    title: str,
    center_rc: Tuple[int, int],
) -> None:
    """matplotlib render with chunk grid overlay + colorbar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    H = stitched_raw.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        stitched_raw,
        cmap="terrain",
        vmin=HEIGHT_GLOBAL_MIN,
        vmax=HEIGHT_GLOBAL_MAX,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Y-level")

    # Chunk gridlines
    # for k in range(1, grid_size):
    #     x = k * CHUNK_SIZE - 0.5
    #     ax.axvline(x, color="white", alpha=0.35, linewidth=0.7)
    #     ax.axhline(x, color="white", alpha=0.35, linewidth=0.7)

    # Mark center chunk with a red outline
    cr, cc = center_rc
    rect = Rectangle(
        (cc * CHUNK_SIZE - 0.5, cr * CHUNK_SIZE - 0.5),
        CHUNK_SIZE, CHUNK_SIZE,
        linewidth=1.5, edgecolor="red", facecolor="none",
    )
    #ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Rendered → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Terrain cLDM inference (Phase 4)")
    p.add_argument("--vae_ckpt",        required=True)
    p.add_argument("--unet_ckpt",       required=True)
    p.add_argument("--controlnet_ckpt", required=True)

    p.add_argument("--prompt",          type=str, default=None,
                   help="Single prompt applied to all chunks")
    p.add_argument("--prompt_north",    type=str, default=None)
    p.add_argument("--prompt_south",    type=str, default=None)
    p.add_argument("--prompt_east",     type=str, default=None)
    p.add_argument("--prompt_west",     type=str, default=None)

    p.add_argument("--grid_size",       type=int, default=3,
                   help="Odd integer (3, 5, 7, ...)")
    p.add_argument("--cfg_scale",       type=float, default=5.0)
    p.add_argument("--num_steps",       type=int, default=50)
    p.add_argument("--seed",            type=int, default=None)

    p.add_argument("--output",          type=str, default="terrain_output.png",
                   help="Path for the matplotlib PNG")
    p.add_argument("--save_npy",        type=str, default=None,
                   help="Optional path for the stitched raw Y-level .npy")

    p.add_argument("--device",          type=str, default=None,
                   help="Override device (default: cuda if available)")

    # ── Java-facing terrain package export ───────────────────────────────────
    p.add_argument("--export_package",  type=str, default=None,
                   help="If set, write a Java-facing terrain_package/ directory here.")
    p.add_argument("--sea_level",       type=int, default=63)
    p.add_argument("--origin_x",        type=int, default=0)
    p.add_argument("--origin_z",        type=int, default=0)
    p.add_argument("--base_y",          type=int, default=0)

    # Smoothing flags — paired --smooth / --no_smooth in argparse style.
    sm = p.add_mutually_exclusive_group()
    sm.add_argument("--smooth",    dest="smooth", action="store_true",
                    help="Enable post-process smoothing (default).")
    sm.add_argument("--no_smooth", dest="smooth", action="store_false",
                    help="Disable all smoothing.")
    p.set_defaults(smooth=True)

    fs = p.add_mutually_exclusive_group()
    fs.add_argument("--feature_smoothing",    dest="feature_smoothing",
                    action="store_true",
                    help="Enable feature-aware smoothing pass (default).")
    fs.add_argument("--no_feature_smoothing", dest="feature_smoothing",
                    action="store_false")
    p.set_defaults(feature_smoothing=True)

    p.add_argument("--smooth_strength", type=float, default=0.3,
                   help="0..1 — smoothing intensity.")

    dec = p.add_mutually_exclusive_group()
    dec.add_argument("--decorations",    dest="decorations", action="store_true",
                     help="Emit decorations.json.gz (default).")
    dec.add_argument("--no_decorations", dest="decorations", action="store_false",
                     help="Emit empty decorations.")
    p.set_defaults(decorations=True)

    args = p.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Validate prompt configuration
    directional = {
        "north": args.prompt_north,
        "south": args.prompt_south,
        "east":  args.prompt_east,
        "west":  args.prompt_west,
    }
    any_directional = any(v is not None for v in directional.values())

    if args.prompt is None and not any_directional:
        raise ValueError("Provide --prompt or at least one of "
                         "--prompt_north/_south/_east/_west.")
    if args.prompt is not None and any_directional:
        print("  [warn] --prompt given alongside --prompt_*; using single --prompt.")
        for k in directional:
            directional[k] = None

    if args.grid_size % 2 == 0:
        raise ValueError(f"grid_size must be odd (got {args.grid_size}).")

    print(f"\n{'='*70}\nPhase 4 inference")
    print(f"{'='*70}")
    print(f"  Device:     {device}")
    print(f"  Grid:       {args.grid_size}x{args.grid_size} "
          f"({args.grid_size * CHUNK_SIZE}x{args.grid_size * CHUNK_SIZE} px)")
    print(f"  cfg_scale:  {args.cfg_scale}")
    print(f"  DDIM steps: {args.num_steps}")
    if args.seed is not None:
        print(f"  Seed:       {args.seed}")

    # ── Load models ──
    vae, unet, controlnet = load_models(
        args.vae_ckpt, args.unet_ckpt, args.controlnet_ckpt, device,
    )
    text_encoder = T5TextEncoder(device=device)
    schedule = CosineNoiseSchedule(num_timesteps=1000)

    # ── Generate ──
    print(f"\n{'='*70}\nGenerating chunks\n{'='*70}")
    chunks, prompts_used = generate_grid(
        grid_size=args.grid_size,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        schedule=schedule,
        text_encoder=text_encoder,
        device=device,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        single_prompt=args.prompt,
        directional=directional,
        seed=args.seed,
    )

    # ── Stitch + denormalize ──
    stitched_norm = stitch_chunks(chunks, args.grid_size)
    stitched_raw = denormalize_height(stitched_norm)

    # ── Seam diagnostics ──
    print(f"\n{'='*70}\nSeam diagnostics (raw Y-levels)\n{'='*70}")
    diag = compute_seam_diagnostics(stitched_raw, args.grid_size)
    print(f"  seams evaluated: {diag['num_seams']}")
    print(f"  mean L1:   {diag['mean_l1']:.2f} Y-levels")
    print(f"  median L1: {diag['median_l1']:.2f}")
    print(f"  max L1:    {diag['max_l1']:.2f}")
    print(f"  horiz mean (between vertically-adjacent chunks): "
          f"{diag['horiz_mean_l1']:.2f}")
    print(f"  vert mean  (between horizontally-adjacent chunks): "
          f"{diag['vert_mean_l1']:.2f}")

    # ── Global stats ──
    print(f"\n{'='*70}\nOutput statistics\n{'='*70}")
    print(f"  Shape:  {stitched_raw.shape}")
    print(f"  Y-min:  {stitched_raw.min():.1f}")
    print(f"  Y-max:  {stitched_raw.max():.1f}")
    print(f"  Y-mean: {stitched_raw.mean():.1f}")

    # ── Save files ──
    title = (
        f"{args.grid_size}×{args.grid_size} grid — "
        f"\"{args.prompt}\"" if args.prompt
        else f"{args.grid_size}×{args.grid_size} grid (directional prompts)"
    )
    render_png(
        stitched_raw,
        grid_size=args.grid_size,
        output_path=args.output,
        title=title,
        center_rc=(args.grid_size // 2, args.grid_size // 2),
    )

    if args.save_npy is not None:
        np.save(args.save_npy, stitched_raw)
        print(f"  Saved raw heightmap → {args.save_npy}")

    # ── Export Java-facing terrain package ────────────────────────────────────
    if args.export_package is not None:
        from terrain_package import build_terrain_package
        print(f"\n{'='*70}\nExporting Java terrain package\n{'='*70}")
        meta = build_terrain_package(
            stitched_raw=stitched_raw,
            prompt=args.prompt,
            directional=directional,
            output_dir=args.export_package,
            origin_x=args.origin_x,
            origin_z=args.origin_z,
            base_y=args.base_y,
            smooth=args.smooth,
            add_materials=True,
            add_decorations=args.decorations,
            sea_level=args.sea_level,
            smooth_strength=args.smooth_strength,
            feature_smoothing=args.feature_smoothing,
            seed=args.seed,
        )
        print(f"  Wrote package → {args.export_package}")
        print(f"    trees:  {meta['counts']['trees']}")
        print(f"    rocks:  {meta['counts']['rocks']}")
        print(f"    plants: {meta['counts']['plants']}")

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
