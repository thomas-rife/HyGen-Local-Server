"""
generate_terrain_img2img.py — Phase 4 inference, img2img edition.

Differences from the original pipeline:

  * The center is no longer a hardcoded flat slab. A procedural macro layer
    upstream produces a *global* macro heightmap covering the entire world
    (in normalized [-1, 1] space, same convention as the rest of the
    pipeline). This file consumes that array.

  * Each AI chunk is initialized as an img2img (SDEdit) refinement of the
    macro slice underneath it: encode the macro slice through the VAE,
    add diffusion noise corresponding to a chosen "start timestep" t_start,
    and DDIM-sample from t_start down to 0 instead of from T-1. This
    preserves the macro shape while letting the AI add detail.

  * Chunks are generated with overlap. World layout is a G x G logical grid
    of 256-pixel chunks placed at stride STEP = 256 - OVERLAP, so adjacent
    chunks share OVERLAP pixels. The model still outputs 256x256; only
    the placement stride changes.

  * Stitching uses a separable cosine (Hann-like) crossfade window over the
    overlap regions. World-edge tiles get half-windows so they don't fade
    against nothing.

The trained VAE / U-Net / ControlNet weights are not modified. Only the
inference loop changes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CHUNK_SIZE      = 256
BORDER_WIDTH    = 16
LATENT_SIZE     = 64
LATENT_CHANNELS = 4
VAE_DOWNSAMPLE  = CHUNK_SIZE // LATENT_SIZE   # = 4

# img2img defaults — tune these from the CLI if you want.
OVERLAP_PX           = 64           # pixel overlap between adjacent chunks
DEFAULT_START_TSTEP  = 600          # 0 = identity, T-1 = full re-gen. ~0.6 keeps macro.
DEFAULT_NUM_STEPS    = 30           # fewer steps OK because we start partway in
DEFAULT_CFG          = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# DDIM SAMPLER — img2img variant
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def ddim_sample_chunk(
    unet,
    controlnet,
    schedule,                          # CosineNoiseSchedule with .alpha_bar (CPU tensor) and .T
    text_emb: torch.Tensor,            # [1, seq_len, 512]
    null_text_emb: torch.Tensor,       # [1, seq_len, 512]
    border_context: torch.Tensor,      # [1, 1, 256, 256] — neighbor border strips
    border_mask: torch.Tensor,         # [1, 1, 256, 256]
    device: str,
    num_steps: int = DEFAULT_NUM_STEPS,
    cfg_scale: float = DEFAULT_CFG,
    init_latent: Optional[torch.Tensor] = None,   # [1, 4, 64, 64], clean (no noise)
    start_timestep: Optional[int] = None,         # 0..T-1; ignored when init_latent is None
) -> torch.Tensor:
    """
    DDIM sampler with optional img2img / SDEdit initialization.

    Behaviour
    ---------
    * If `init_latent is None`: identical to the original pure-noise path
      (start at t=T-1, denoise to 0 over `num_steps`).
    * If `init_latent is not None` and `start_timestep is not None`:
        z_t = sqrt(alpha_bar[t_start]) * init_latent
            + sqrt(1 - alpha_bar[t_start]) * noise
      and DDIM denoises from `start_timestep` down to 0 over `num_steps`.

    The ControlNet border path is unchanged — img2img handles macro
    structure, border context still handles edge matching with previously
    generated neighbors.
    """
    unet.eval()
    controlnet.eval()

    T = schedule.T
    amp_device = device.split(":")[0]
    use_amp = device.startswith("cuda")

    # ── Build the timestep schedule we'll iterate over ─────────────────────
    if init_latent is not None and start_timestep is not None:
        t_hi = int(max(1, min(T - 1, start_timestep)))
        step_indices = torch.linspace(t_hi, 0, num_steps, dtype=torch.long)

        # Diffuse the clean init latent to t_hi:  q(z_t | z_0)
        ab_start = schedule.alpha_bar[t_hi].to(device).view(1, 1, 1, 1)
        noise = torch.randn_like(init_latent)
        z = ab_start.sqrt() * init_latent + (1.0 - ab_start).sqrt() * noise
    else:
        # Original behaviour: pure-noise start.
        step_indices = torch.linspace(T - 1, 0, num_steps, dtype=torch.long)
        z = torch.randn(1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device=device)

    # ── DDIM loop ──────────────────────────────────────────────────────────
    for i, t_val in enumerate(step_indices):
        t = torch.full((1,), int(t_val.item()), device=device, dtype=torch.long)

        # Conditional pass
        if use_amp:
            with autocast(device_type=amp_device, dtype=torch.float16):
                residuals_c = controlnet(z, t, text_emb, border_context, border_mask)
                eps_cond = unet(z, t, text_emb, controlnet_residuals=residuals_c)
        else:
            residuals_c = controlnet(z, t, text_emb, border_context, border_mask)
            eps_cond = unet(z, t, text_emb, controlnet_residuals=residuals_c)

        # Unconditional pass for CFG
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

        # DDIM update
        ab_t = schedule.alpha_bar[t.cpu()].to(device)
        while ab_t.dim() < z.dim():
            ab_t = ab_t.unsqueeze(-1)

        if i < len(step_indices) - 1:
            t_prev = step_indices[i + 1]              # already on CPU
            ab_prev = schedule.alpha_bar[t_prev].to(device)
            while ab_prev.dim() < z.dim():
                ab_prev = ab_prev.unsqueeze(-1)
        else:
            ab_prev = torch.ones_like(ab_t)

        x0_pred = (z - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
        x0_pred = torch.clamp(x0_pred, -3.0, 3.0)
        dir_zt  = (1.0 - ab_prev).sqrt() * eps
        z       = ab_prev.sqrt() * x0_pred + dir_zt

    return z


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAPPING-CHUNK LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

def compute_chunk_origins(grid_size: int, overlap: int = OVERLAP_PX
                          ) -> Tuple[List[int], int, int]:
    """
    Return the list of pixel origins (top-left) for a 1-D row of `grid_size`
    chunks, the per-chunk stride, and the total world extent in pixels.

    Layout: chunks are CHUNK_SIZE wide, placed at stride = CHUNK_SIZE - overlap.
    Total world size = stride * (grid_size - 1) + CHUNK_SIZE.
    """
    assert 0 <= overlap < CHUNK_SIZE, f"overlap must be in [0, {CHUNK_SIZE})"
    stride = CHUNK_SIZE - overlap
    origins = [k * stride for k in range(grid_size)]
    world_size = stride * (grid_size - 1) + CHUNK_SIZE
    return origins, stride, world_size


def slice_macro_for_chunk(
    macro_world: np.ndarray,            # [world_size, world_size] float, normalized [-1, 1]
    row: int, col: int,
    origins_y: List[int], origins_x: List[int],
) -> np.ndarray:
    """Return a [CHUNK_SIZE, CHUNK_SIZE] crop of the macro for this chunk."""
    y0 = origins_y[row]
    x0 = origins_x[col]
    return macro_world[y0:y0 + CHUNK_SIZE, x0:x0 + CHUNK_SIZE]


# ══════════════════════════════════════════════════════════════════════════════
# BORDER CONTEXT — overlap-aware
# ══════════════════════════════════════════════════════════════════════════════

def build_border_context_overlap(
    chunk_rc: Tuple[int, int],
    chunks: Dict[Tuple[int, int], torch.Tensor],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Same conditioning convention as the original: pull a BORDER_WIDTH strip
    from each already-generated orthogonal neighbor. We still take the strip
    from the neighbor's edge that *physically faces* this chunk — with
    overlap, that strip happens to lie inside the overlap region of both
    chunks, which is exactly what we want (it nudges this chunk to agree
    with the neighbor in the seam zone).

    Returns (ctx [1,1,256,256], mask [1,1,256,256], list_of_active_edges).
    """
    r, c = chunk_rc
    bw, cs = BORDER_WIDTH, CHUNK_SIZE

    ctx  = torch.zeros(1, 1, cs, cs, device=device)
    mask = torch.zeros(1, 1, cs, cs, device=device)
    active: List[str] = []

    above = chunks.get((r - 1, c))
    if above is not None:
        ctx[:, :, :bw, :]  = above[:, :, -bw:, :].clamp(-1.0, 1.0)
        mask[:, :, :bw, :] = 1.0
        active.append("top")

    below = chunks.get((r + 1, c))
    if below is not None:
        ctx[:, :, -bw:, :]  = below[:, :, :bw, :].clamp(-1.0, 1.0)
        mask[:, :, -bw:, :] = 1.0
        active.append("bottom")

    left = chunks.get((r, c - 1))
    if left is not None:
        ctx[:, :, :, :bw]  = left[:, :, :, -bw:].clamp(-1.0, 1.0)
        mask[:, :, :, :bw] = 1.0
        active.append("left")

    right = chunks.get((r, c + 1))
    if right is not None:
        ctx[:, :, :, -bw:]  = right[:, :, :, :bw].clamp(-1.0, 1.0)
        mask[:, :, :, -bw:] = 1.0
        active.append("right")

    return ctx, mask, active


# ══════════════════════════════════════════════════════════════════════════════
# GRID GENERATION — img2img driven
# ══════════════════════════════════════════════════════════════════════════════

def generate_grid(
    grid_size: int,
    macro_world: np.ndarray,            # [world_size, world_size] normalized macro heightmap
    vae,
    unet,
    controlnet,
    schedule,
    text_encoder,
    device: str,
    cfg_scale: float = DEFAULT_CFG,
    num_steps: int = DEFAULT_NUM_STEPS,
    single_prompt: Optional[str] = None,
    directional: Optional[Dict[str, Optional[str]]] = None,
    seed: Optional[int] = None,
    overlap: int = OVERLAP_PX,
    start_timestep: int = DEFAULT_START_TSTEP,
    img2img_strength: Optional[float] = None,    # if set in [0,1], overrides start_timestep
    pick_prompt_for_chunk_fn=None,               # injected from caller; same signature as before
    spiral_order_fn=None,                        # injected from caller
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], List[str], List[int], List[int]]:
    """
    Generate a G x G grid of overlapping chunks via img2img refinement of
    `macro_world`. Returns:

        chunks_norm   : {(row, col): [1, 1, 256, 256] normalized tensor}
        prompts_used  : prompt string per generated chunk (in spiral order)
        origins_y     : list of row-origin pixels (length grid_size)
        origins_x     : list of col-origin pixels

    No "center is special / hardcoded flat" branch any more — every chunk
    runs the same img2img path. The macro layer decides what's at the
    center (a bowl, a canyon floor, a coastline, etc.). If the caller
    *does* want the center to be its own thing (e.g. raw procedural with
    no AI), they can substitute it post-hoc by overwriting chunks[(cr,cc)].

    Notes on `start_timestep` vs `img2img_strength`
    -----------------------------------------------
    Following SDEdit / Stable Diffusion img2img convention:
        strength s in [0, 1]   maps to   start_timestep = round(s * (T - 1))
    s = 0.0 → identity (no change), s = 1.0 → pure noise (ignore macro).
    Default 600 / 1000 ≈ 0.6, which is a strong-but-not-destructive refine.
    """
    assert grid_size >= 1
    if directional is None:
        directional = {}

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)

    # Resolve start timestep
    if img2img_strength is not None:
        s = float(max(0.0, min(1.0, img2img_strength)))
        t_start = int(round(s * (schedule.T - 1)))
    else:
        t_start = int(start_timestep)

    # Layout
    origins_x, stride, world_size = compute_chunk_origins(grid_size, overlap)
    origins_y = list(origins_x)  # square grid
    expected = stride * (grid_size - 1) + CHUNK_SIZE
    if macro_world.shape != (expected, expected):
        raise ValueError(
            f"macro_world shape {macro_world.shape} does not match "
            f"expected ({expected}, {expected}) for grid={grid_size}, "
            f"overlap={overlap}. Procedural macro must be sized to match."
        )

    # Pre-encode the whole macro to latent once (we'll slice latents per chunk).
    # This is cheaper and avoids edge artifacts from per-chunk VAE encodes
    # whose receptive fields would otherwise truncate at chunk boundaries.
    macro_t = (
        torch.from_numpy(macro_world.astype(np.float32))
        .unsqueeze(0).unsqueeze(0).to(device)            # [1, 1, H, H]
        .clamp(-1.0, 1.0)
    )
    use_amp = device.startswith("cuda")
    amp_device = device.split(":")[0]
    with torch.inference_mode():
        if use_amp:
            with autocast(device_type=amp_device, dtype=torch.float16):
                macro_latent_full, _ = vae.encode(macro_t)
                macro_latent_full = macro_latent_full.float()
        else:
            macro_latent_full, _ = vae.encode(macro_t)
    # macro_latent_full: [1, 4, world_size/4, world_size/4]

    chunks: Dict[Tuple[int, int], torch.Tensor] = {}
    prompts_used: List[str] = []

    # Generation order
    if spiral_order_fn is not None:
        order = spiral_order_fn(grid_size)
        # spiral_order in the original code excludes the center; for img2img
        # we *do* want the center — append it last so it benefits from
        # neighbor borders on every side.
        cr = cc = grid_size // 2
        if (cr, cc) not in order:
            order = order + [(cr, cc)]
    else:
        # Fallback: simple raster order.
        order = [(r, c) for r in range(grid_size) for c in range(grid_size)]

    null_emb = text_encoder.null_embedding
    total = len(order)

    for i, rc in enumerate(order, start=1):
        r, c = rc

        # Prompt for this cell
        if pick_prompt_for_chunk_fn is not None:
            prompt = pick_prompt_for_chunk_fn(rc, grid_size, single_prompt, directional)
        else:
            prompt = single_prompt or "varied natural terrain"
        prompts_used.append(prompt)

        text_emb = text_encoder.encode(prompt)

        # Border context from already-generated neighbors
        border_ctx, border_mask, active_edges = build_border_context_overlap(
            rc, chunks, device,
        )

        # Slice the precomputed macro latent for this chunk
        ly = origins_y[r] // VAE_DOWNSAMPLE
        lx = origins_x[c] // VAE_DOWNSAMPLE
        init_latent = macro_latent_full[
            :, :, ly:ly + LATENT_SIZE, lx:lx + LATENT_SIZE
        ].contiguous()

        print(f"  [{i:2d}/{total}] ({r},{c})  "
              f"borders: {active_edges or ['none']}  "
              f"t_start: {t_start}  prompt: \"{prompt[:48]}\"")

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
            init_latent=init_latent,
            start_timestep=t_start,
        )

        # Decode to heightmap space
        if use_amp:
            with autocast(device_type=amp_device, dtype=torch.float16):
                hm = vae.decode(z_final).float()
        else:
            hm = vae.decode(z_final)
        hm = hm.clamp(-1.0, 1.0)

        chunks[rc] = hm

        del z_final
        if use_amp:
            torch.cuda.empty_cache()

    return chunks, prompts_used, origins_y, origins_x


# ══════════════════════════════════════════════════════════════════════════════
# CROSSFADE STITCHING
# ══════════════════════════════════════════════════════════════════════════════

def _hann_ramp(length: int) -> np.ndarray:
    """
    Cosine ramp 0 → 1 over `length` samples, inclusive at both ends in the
    sense that the curve hits 0 at index 0 and 1 at index length-1. We use
    a half-Hann (1 - cos) / 2 — symmetric, smooth derivative at both ends,
    which is what "cosine crossfade" usually means.
    """
    if length <= 1:
        return np.ones(max(1, length), dtype=np.float32)
    n = np.arange(length, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(np.pi * n / (length - 1)))


def build_chunk_window(
    overlap: int,
    fade_top: bool, fade_bottom: bool,
    fade_left: bool, fade_right: bool,
) -> np.ndarray:
    """
    Build a [CHUNK_SIZE, CHUNK_SIZE] separable cosine window. The window is
    1.0 in the chunk interior and ramps down to 0 over `overlap` pixels on
    each side that has a neighbor (i.e. that side fades). World-edge sides
    are NOT faded (weight stays 1.0 there) so the world edge isn't
    artificially attenuated.
    """
    cs = CHUNK_SIZE
    if overlap <= 0:
        return np.ones((cs, cs), dtype=np.float32)

    # 1-D weights along one axis
    def axis_weights(fade_lo: bool, fade_hi: bool) -> np.ndarray:
        w = np.ones(cs, dtype=np.float32)
        if fade_lo:
            w[:overlap] = _hann_ramp(overlap)              # 0 → 1 going inward
        if fade_hi:
            w[cs - overlap:] = _hann_ramp(overlap)[::-1]   # 1 → 0 going outward
        return w

    wx = axis_weights(fade_left, fade_right)
    wy = axis_weights(fade_top, fade_bottom)
    return np.outer(wy, wx).astype(np.float32)             # [cs, cs]


def stitch_chunks_crossfade(
    chunks: Dict[Tuple[int, int], torch.Tensor],
    grid_size: int,
    origins_y: List[int],
    origins_x: List[int],
    overlap: int = OVERLAP_PX,
) -> np.ndarray:
    """
    Weighted-average overlapping chunks into a single normalized heightmap.

    For every pixel (y, x) in the world:
        out[y, x] = sum_k(w_k * h_k) / sum_k(w_k)
    where the sum is over chunks covering that pixel and w_k is the chunk's
    cosine window. Chunk sides facing a real neighbor fade; sides at the
    world boundary do not.
    """
    if grid_size <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    stride = CHUNK_SIZE - overlap
    world_size = stride * (grid_size - 1) + CHUNK_SIZE

    accum = np.zeros((world_size, world_size), dtype=np.float32)
    weight = np.zeros((world_size, world_size), dtype=np.float32)

    for (r, c), hm_t in chunks.items():
        patch = hm_t.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Decide which sides fade based on whether a neighbor exists in-grid
        fade_top    = (r > 0)
        fade_bottom = (r < grid_size - 1)
        fade_left   = (c > 0)
        fade_right  = (c < grid_size - 1)

        win = build_chunk_window(
            overlap=overlap,
            fade_top=fade_top,
            fade_bottom=fade_bottom,
            fade_left=fade_left,
            fade_right=fade_right,
        )

        y0, x0 = origins_y[r], origins_x[c]
        y1, x1 = y0 + CHUNK_SIZE, x0 + CHUNK_SIZE

        accum[y0:y1, x0:x1]  += patch * win
        weight[y0:y1, x0:x1] += win

    # Anywhere weight is exactly zero (shouldn't happen with a valid layout)
    # we fall back to zero rather than NaN.
    safe = weight > 1e-8
    out = np.zeros_like(accum)
    out[safe] = accum[safe] / weight[safe]

    return out
