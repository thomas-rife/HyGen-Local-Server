"""
terrain_package.py — Java-facing terrain package export.

This module takes a stitched raw heightmap produced by the diffusion pipeline
(see generate_terrain.py) and writes a self-contained `terrain_package/`
directory that a Java mod (Hytale / Odyssey) can download and render.

The package layout is:

    terrain_package/
      metadata.json
      heightmap.bin.gz       uint16 little-endian, shape [depth, width]
      materialmap.bin.gz     uint8 material IDs, shape [depth, width]
      watermap.bin.gz        uint8 water mask (0 = none, 1 = water)
      decorations.json.gz    high-level decoration intents (trees, rocks, plants)
      preview.png            matplotlib render for debugging

Design contract (do NOT break this):
    Python owns:  smoothing, material classification, water/river decisions,
                  decoration candidate positions.
    Java owns:    exact game block IDs, placement, batching, collision checks,
                  prefab choice for trees/rocks.

Python must NEVER emit Hytale block IDs. We emit generic material/decor IDs
only. Java maps those to whatever its current block palette is.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — material IDs, Y-level clamping, defaults
# ══════════════════════════════════════════════════════════════════════════════

MATERIAL_IDS: Dict[int, str] = {
    0: "grass",
    1: "dirt",
    2: "stone",
    3: "sand",
    4: "snow",
    5: "rock",
    6: "mud",
    7: "gravel",
    8: "water_floor",
}
NAME_TO_MATERIAL: Dict[str, int] = {v: k for k, v in MATERIAL_IDS.items()}

# Height range Java cares about. Matches the training data's global stats —
# see terrain_dataloader.HEIGHT_GLOBAL_MIN/MAX. We clamp into [0, 65535] at
# export time because heightmap.bin is uint16.
Y_CLAMP_MIN = 0
Y_CLAMP_MAX = 511  # Hytale-like vertical range; still fits trivially in uint16

DEFAULT_SEA_LEVEL = 63
PACKAGE_VERSION = 1


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT PARSING
# ══════════════════════════════════════════════════════════════════════════════

# Keyword buckets drive material/water/decoration decisions. These are the
# categories listed in the spec — do not silently add new ones without also
# wiring them through the classifiers below.
PROMPT_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "desert":   ("desert", "dunes", "dune", "oasis", "arid", "sandstone", "sahara"),
    "snow":     ("snow", "snowy", "alpine", "frozen", "tundra", "ice", "glacier",
                 "arctic"),
    "mountain": ("mountain", "mountains", "cliff", "cliffs", "canyon", "rocky",
                 "peak", "peaks", "crag"),
    "forest":   ("forest", "woods", "jungle", "taiga", "woodland"),
    "plains":   ("plains", "plain", "meadow", "grassland", "steppe", "valley"),
    "beach":    ("beach", "coast", "coastline", "island", "shore", "shoreline"),
    "swamp":    ("swamp", "marsh", "bog", "mire", "wetland"),
    "water":    ("ocean", "sea", "lake", "river", "coast", "coastline", "island",
                 "swamp", "marsh", "beach"),
    "ocean":    ("ocean", "sea", "deep water"),
    "lake":     ("lake", "loch", "tarn", "pond", "reservoir"),
    "river":    ("river", "stream", "creek"),
    "tropical": ("tropical", "jungle", "palm", "island"),
}


def _prompt_text(prompt: Optional[str], directional: Optional[Dict[str, Optional[str]]]) -> str:
    """Flatten single prompt + any directional prompts into one lowercase blob.

    The classifiers below only ever ask "does this keyword appear" so we don't
    care about preserving structure — we just want every keyword the user
    typed anywhere to be visible.
    """
    parts: List[str] = []
    if prompt:
        parts.append(prompt)
    if directional:
        for v in directional.values():
            if v:
                parts.append(v)
    return " ".join(parts).lower()


def _has_any(text: str, keywords: Tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


@dataclass
class PromptFlags:
    """Simple boolean summary of the prompt used by every downstream step."""
    desert:   bool
    snow:     bool
    mountain: bool
    forest:   bool
    plains:   bool
    beach:    bool
    swamp:    bool
    ocean:    bool
    river:    bool
    lake:     bool
    tropical: bool

    @property
    def has_water(self) -> bool:
        # "water" in the spec covers ocean / lake / river / coast / island /
        # swamp — treat any of those as "this map wants water somewhere".
        return self.ocean or self.river or self.beach or self.swamp or self.lake


def classify_prompt(prompt: Optional[str],
                    directional: Optional[Dict[str, Optional[str]]]) -> PromptFlags:
    text = _prompt_text(prompt, directional)
    return PromptFlags(
        desert=_has_any(text, PROMPT_KEYWORDS["desert"]),
        snow=_has_any(text, PROMPT_KEYWORDS["snow"]),
        mountain=_has_any(text, PROMPT_KEYWORDS["mountain"]),
        forest=_has_any(text, PROMPT_KEYWORDS["forest"]),
        plains=_has_any(text, PROMPT_KEYWORDS["plains"]),
        beach=_has_any(text, PROMPT_KEYWORDS["beach"]),
        swamp=_has_any(text, PROMPT_KEYWORDS["swamp"]),
        ocean=_has_any(text, PROMPT_KEYWORDS["ocean"]),
        river=_has_any(text, PROMPT_KEYWORDS["river"]),
        lake=_has_any(text, PROMPT_KEYWORDS["lake"]),
        tropical=_has_any(text, PROMPT_KEYWORDS["tropical"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING  —  two-stage post-process before export
# ══════════════════════════════════════════════════════════════════════════════
#
# Stage 1 (micro):  small Gaussian / box blur to kill single-pixel spikes.
#                   Always on by default. Strength is controllable.
# Stage 2 (feature-aware):  extra smoothing only where it helps gameplay —
#                   near water edges, along rivers, on beach/desert terrain.
#                   Mountains are preserved (we actively MASK OUT high-slope
#                   pixels so peaks stay sharp).

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    return k / k.sum()


def _separable_gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 1D Gaussian blur on a 2D float array.

    Uses `np.pad(..., mode='edge')` so edges don't drift to zero — important
    because this is called on the *full* stitched map, not per-chunk, and we
    want the borders to stay consistent with their neighbors.
    """
    if sigma <= 0.0:
        return arr
    radius = max(1, int(math.ceil(sigma * 3.0)))
    k = _gaussian_kernel_1d(sigma, radius)

    padded = np.pad(arr, ((0, 0), (radius, radius)), mode="edge")
    # Horizontal pass
    blurred = np.zeros_like(arr)
    for i, w in enumerate(k):
        blurred += w * padded[:, i : i + arr.shape[1]]

    padded = np.pad(blurred, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for i, w in enumerate(k):
        out += w * padded[i : i + arr.shape[0], :]
    return out


def _slope_magnitude(height: np.ndarray) -> np.ndarray:
    """Per-pixel slope magnitude via central differences. Same shape as input."""
    gy = np.zeros_like(height)
    gx = np.zeros_like(height)
    gy[1:-1, :] = (height[2:, :] - height[:-2, :]) * 0.5
    gx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def smooth_heightmap(
    stitched_raw: np.ndarray,
    flags: PromptFlags,
    smooth: bool = True,
    strength: float = 0.3,
    feature_smoothing: bool = True,
) -> np.ndarray:
    """Two-stage post-process per the spec.

    Parameters
    ----------
    stitched_raw : float or int heightmap, shape [depth, width].
    flags        : prompt classification.
    smooth       : enable the micro pass. If False the whole function is a no-op.
    strength     : 0..1 — controls both Gaussian sigma and how aggressively
                   the feature-aware pass blends in. 0 = identity, 1 = strong.
    feature_smoothing : enable the feature-aware second pass.

    Returns
    -------
    A float32 array of the same shape, still in raw Y-level units. Clamped to
    [Y_CLAMP_MIN, Y_CLAMP_MAX].
    """
    h = stitched_raw.astype(np.float32, copy=True)
    if not smooth:
        return np.clip(h, Y_CLAMP_MIN, Y_CLAMP_MAX)

    strength = float(max(0.0, min(1.0, strength)))

    # ── Stage 1: micro Gaussian ──────────────────────────────────────────────
    # sigma ~ strength * 1.2 keeps us in the 3-to-5-pixel kernel range at
    # default strength 0.3. We apply a slope-based mask so peaks are preserved
    # even in the "always on" pass: if the local slope is large, we blend the
    # original back in.
    sigma = 0.4 + strength * 1.2
    blurred = _separable_gaussian(h, sigma)

    slope = _slope_magnitude(h)
    # Normalise slope into [0, 1]. Slopes > ~6 Y/pixel count as "steep" — on
    # our training scale a cliff face is easily 10-20 Y/pixel.
    slope_norm = np.clip(slope / 6.0, 0.0, 1.0)
    # Low-slope regions get 100% blur; high-slope regions keep more original.
    blend = (1.0 - slope_norm)
    h = h * (1.0 - blend) + blurred * blend

    # ── Stage 2: feature-aware ───────────────────────────────────────────────
    if feature_smoothing and strength > 0.0:
        if flags.has_water:
            # Smooth low-elevation areas harder so the waterline doesn't have
            # ragged pixel cliffs. Identify "low" as within ~8 Y of sea level.
            near_water_band = np.clip(
                1.0 - np.abs(h - DEFAULT_SEA_LEVEL) / 8.0, 0.0, 1.0
            )
            extra = _separable_gaussian(h, 1.2 + strength * 1.5)
            # Don't over-smooth cliffs even near water unless the prompt is
            # specifically a coast/beach with no cliff language.
            cliff_guard = 1.0 - slope_norm * (0.0 if flags.beach else 0.5)
            w = near_water_band * cliff_guard * (0.5 + 0.5 * strength)
            h = h * (1.0 - w) + extra * w

        if flags.river:
            # Rivers want a smoother continuous path. _carve_rivers does the
            # actual carving; here we just pre-smooth so banks don't look
            # bit-crushed around the future riverbed.
            extra = _separable_gaussian(h, 1.5 + strength)
            h = h * 0.65 + extra * 0.35

        if flags.beach:
            # Flatten the shoreline slightly — compress mid-slopes near
            # sea level. Same band mask as the water case, but heavier blur.
            near_water_band = np.clip(
                1.0 - np.abs(h - DEFAULT_SEA_LEVEL) / 12.0, 0.0, 1.0
            )
            extra = _separable_gaussian(h, 2.0 + strength * 1.5)
            w = near_water_band * (0.6 + 0.4 * strength)
            h = h * (1.0 - w) + extra * w

        if flags.desert:
            # Reduce high-frequency noise across the whole map — dunes should
            # look rolling, not pixelated.
            extra = _separable_gaussian(h, 1.8 + strength * 1.2)
            w = 0.5 * (0.5 + 0.5 * strength)
            h = h * (1.0 - w) + extra * w

        if flags.mountain:
            # Guard: mountain prompts explicitly want sharp peaks. Blend the
            # *original unsmoothed* back in wherever slope is high. This also
            # partially undoes the micro pass for those pixels, which is
            # intended — "mountain" is the user asking for detail.
            orig = stitched_raw.astype(np.float32)
            h = h * (1.0 - slope_norm) + orig * slope_norm

    return np.clip(h, Y_CLAMP_MIN, Y_CLAMP_MAX)


# ══════════════════════════════════════════════════════════════════════════════
# WATERMAP  —  oceans, lakes, coastlines, and river carving
# ══════════════════════════════════════════════════════════════════════════════

def _carve_rivers(
    height: np.ndarray,
    water: np.ndarray,
    sea_level: int,
    num_rivers: int,
    rng: random.Random,
) -> None:
    """Carve terrain-aware descending river corridors.

    MUTATES `height` and `water` in place. Each river starts on a random edge
    and walks toward the opposite edge, jittered by noise, carving a shallow
    valley ~5 pixels wide and marking the centre line as water. This is not
    hydrologically accurate — it just needs to look like "a river goes through
    here" from above.
    """
    depth, width = height.shape
    if depth < 16 or width < 16:
        return

    for river_index in range(max(1, int(num_rivers))):
        # Choose a broad crossing direction.
        side = rng.choice(("lr", "tb"))

        points: List[Tuple[float, float]] = []

        if side == "lr":
            start_r = rng.uniform(depth * 0.25, depth * 0.75)
            end_r = rng.uniform(depth * 0.25, depth * 0.75)

            # Control points give a broad meander.
            c0 = 0.0
            c3 = float(width - 1)
            c1 = width * rng.uniform(0.25, 0.40)
            c2 = width * rng.uniform(0.60, 0.75)

            r0 = start_r
            r3 = end_r
            r1 = np.clip(start_r + rng.uniform(-depth * 0.22, depth * 0.22), 2, depth - 3)
            r2 = np.clip(end_r + rng.uniform(-depth * 0.22, depth * 0.22), 2, depth - 3)

            samples = max(width, depth)
            for i in range(samples):
                t = i / max(samples - 1, 1)
                u = 1.0 - t
                c = (u ** 3) * c0 + 3 * (u ** 2) * t * c1 + 3 * u * (t ** 2) * c2 + (t ** 3) * c3
                r = (u ** 3) * r0 + 3 * (u ** 2) * t * r1 + 3 * u * (t ** 2) * r2 + (t ** 3) * r3
                points.append((r, c))
        else:
            start_c = rng.uniform(width * 0.25, width * 0.75)
            end_c = rng.uniform(width * 0.25, width * 0.75)

            r0 = 0.0
            r3 = float(depth - 1)
            r1 = depth * rng.uniform(0.25, 0.40)
            r2 = depth * rng.uniform(0.60, 0.75)

            c0 = start_c
            c3 = end_c
            c1 = np.clip(start_c + rng.uniform(-width * 0.22, width * 0.22), 2, width - 3)
            c2 = np.clip(end_c + rng.uniform(-width * 0.22, width * 0.22), 2, width - 3)

            samples = max(width, depth)
            for i in range(samples):
                t = i / max(samples - 1, 1)
                u = 1.0 - t
                r = (u ** 3) * r0 + 3 * (u ** 2) * t * r1 + 3 * u * (t ** 2) * r2 + (t ** 3) * r3
                c = (u ** 3) * c0 + 3 * (u ** 2) * t * c1 + 3 * u * (t ** 2) * c2 + (t ** 3) * c3
                points.append((r, c))

        if not points:
            continue

        # Sample terrain along path.
        sampled = []
        for r_f, c_f in points:
            r_i = int(np.clip(round(r_f), 0, depth - 1))
            c_i = int(np.clip(round(c_f), 0, width - 1))
            sampled.append(float(height[r_i, c_i]))

        source_h = float(np.percentile(sampled[: max(8, len(sampled) // 8)], 65))
        mouth_h = float(np.percentile(sampled[-max(8, len(sampled) // 8):], 35))

        # Keep the profile sane. River should descend, but not become an absurd canyon.
        high = max(source_h, mouth_h)
        low = min(source_h, mouth_h)
        terrain_range = float(np.percentile(height, 95) - np.percentile(height, 5))

        # Prefer an outlet near sea level if the terrain supports it.
        target_mouth = min(low - 0.5, float(sea_level) - 1.5)
        if terrain_range < 35:
            target_source = min(high - 0.75, target_mouth + 3.0)
        elif terrain_range < 80:
            target_source = max(high - 2.0, target_mouth + 4.5)
        else:
            target_source = max(high - 3.5, target_mouth + 6.0)

        # If the whole terrain is low/flat, use a shallow profile near sea level.
        if target_source - target_mouth < 4.0:
            if terrain_range < 35:
                target_mouth = float(sea_level) - 1.25
                target_source = target_mouth + 2.5
            else:
                target_mouth = float(sea_level) - 2.0
                target_source = target_mouth + 4.0

        target_mouth = max(float(Y_CLAMP_MIN), target_mouth)
        target_source = max(target_mouth + 2.0, target_source)

        # Widths in pixels. Plains get wider, softer rivers. Mountains remain tighter.
        if terrain_range < 35:
            flood_radius = 14
            bank_radius = 7
            water_radius = 3
            carve_strength = 0.48
        elif terrain_range < 80:
            flood_radius = 10
            bank_radius = 5
            water_radius = 2
            carve_strength = 0.72
        else:
            flood_radius = 6
            bank_radius = 4
            water_radius = 1
            carve_strength = 0.90

        for i, (r_f, c_f) in enumerate(points):
            t = i / max(len(points) - 1, 1)

            # Smooth descending profile from source to mouth.
            bed = (1.0 - t) * target_source + t * target_mouth

            r = int(np.clip(round(r_f), 0, depth - 1))
            c = int(np.clip(round(c_f), 0, width - 1))

            rr0 = max(0, r - flood_radius)
            rr1 = min(depth, r + flood_radius + 1)
            cc0 = max(0, c - flood_radius)
            cc1 = min(width, c + flood_radius + 1)

            yy, xx = np.ogrid[rr0:rr1, cc0:cc1]
            dist = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)

            local = height[rr0:rr1, cc0:cc1]

            # Inner wet channel.
            channel_mask = dist <= water_radius

            # Bank cuts toward the river bed.
            bank_mask = dist <= bank_radius
            bank_weight = np.clip(1.0 - (dist / max(bank_radius, 1)), 0.0, 1.0)

            # Floodplain gently lowers/smooths nearby terrain without making a trench.
            flood_mask = dist <= flood_radius
            flood_weight = np.clip(1.0 - (dist / max(flood_radius, 1)), 0.0, 1.0) ** 2

            # Target cross-section:
            # center is bed, banks are slightly above bed, outer floodplain is subtle.
            bank_target = bed + 2.0 + dist * 0.35
            flood_target = bed + 5.0 + dist * 0.45

            lowered = local.copy()

            # Only lower terrain, never raise terrain here.
            bank_new = local * (1.0 - bank_weight * carve_strength) + bank_target * (bank_weight * carve_strength)
            flood_new = local * (1.0 - flood_weight * 0.35) + flood_target * (flood_weight * 0.35)

            lowered = np.where(flood_mask & (flood_new < lowered), flood_new, lowered)
            lowered = np.where(bank_mask & (bank_new < lowered), bank_new, lowered)

            # Force inner channel just below water surface.
            lowered = np.where(channel_mask & (bed < lowered), bed, lowered)

            height[rr0:rr1, cc0:cc1] = lowered

            # Mark water only in the inner channel.
            water[rr0:rr1, cc0:cc1] = np.where(channel_mask, 1, water[rr0:rr1, cc0:cc1])


def _identify_lake_islands(
    height: np.ndarray,
    water: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    from scipy import ndimage

    water_bool = water.astype(bool)
    if not np.any(water_bool):
        return np.zeros_like(water_bool)

    land = (~water_bool) & (height > float(sea_level) + 1.0)
    if not np.any(land):
        return np.zeros_like(water_bool)

    labels, count = ndimage.label(land)
    islands = np.zeros_like(water_bool)

    for idx in range(1, count + 1):
        comp = labels == idx
        area = int(np.sum(comp))
        if area < 12 or area > 6000:
            continue

        rows, cols = np.where(comp)
        if rows.size == 0:
            continue
        if (
            rows.min() == 0 or rows.max() == height.shape[0] - 1
            or cols.min() == 0 or cols.max() == height.shape[1] - 1
        ):
            continue

        boundary = ndimage.binary_dilation(comp, iterations=1) & ~comp
        if not np.any(boundary):
            continue
        if float(np.mean(water_bool[boundary])) < 0.65:
            continue

        filled = ndimage.binary_fill_holes(comp)
        if int(np.sum(filled)) > 8000:
            continue
        islands |= filled

    return islands


def _preserve_lake_islands(
    height: np.ndarray,
    water: np.ndarray,
    sea_level: int,
) -> np.ndarray:
    islands = _identify_lake_islands(height, water, sea_level)
    if not np.any(islands):
        return islands

    fill_height = np.maximum(height[islands], float(sea_level) + 2.0)
    if np.any(fill_height):
        median_height = float(np.median(fill_height))
        height[islands] = np.maximum(height[islands], max(median_height, float(sea_level) + 2.0))
    water[islands] = 0
    return islands


def build_watermap(
    height: np.ndarray,
    flags: PromptFlags,
    sea_level: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the water mask and optionally modify `height` to carve rivers.

    Returns
    -------
    watermap : uint8 [depth, width], 0 = dry, 1 = water.
    height_out : possibly modified copy of `height` (only modified for rivers).
    """
    h = height.astype(np.float32, copy=True)
    water = np.zeros(h.shape, dtype=np.uint8)

    # Oceans/coasts use global sea-level flooding.
    if flags.ocean or flags.beach:
        water[h < sea_level] = 1

    # Lakes should not behave like oceans. For lake prompts, fill only the
    # lowest basin-like cells near/below sea level so the entire center does
    # not become a generic ocean flood.
    if flags.lake and not (flags.ocean or flags.beach):
        lake_level = float(sea_level)
        low = h < (lake_level - 0.5)

        if np.any(low):
            from scipy import ndimage

            labels, count = ndimage.label(low)
            if count > 0:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                largest = int(np.argmax(sizes))

                lake_mask = labels == largest

                # Slightly expand the lake to get a smoother shoreline, but
                # do not flood unrelated low pockets.
                lake_mask = ndimage.binary_dilation(lake_mask, iterations=1)
                lake_mask &= h < (lake_level + 0.5)

                water[lake_mask] = 1

                # Pull lake bed down slightly where water exists so the lake
                # has a floor and does not z-fight with shore.
                h[lake_mask] = np.minimum(h[lake_mask], lake_level - 2.0)
                _preserve_lake_islands(h, water, sea_level)

    # Swamps: shallow water on the very lowest cells only. We lower cells a
    # few Y below sea level in low bowls so Java still has a floor to place
    # "water_floor" material on.
    if flags.swamp:
        # low-20% of the height distribution → shallow pools
        thresh = float(np.percentile(h, 20))
        swamp_mask = h <= thresh
        # don't submerge everything — stay within a couple Y of sea level
        water[swamp_mask & (h < sea_level + 1)] = 1

    # Rivers: carve up to 2 meandering paths.
    if flags.river:
        rng = random.Random(seed if seed is not None else 0xC0FFEE)
        
        # Respect prompt intent when possible.
        # Small worlds get one river; larger worlds can support two.
        num_rivers = 1 if min(h.shape) < 400 else 2

        _carve_rivers(h, water, sea_level, num_rivers, rng)

        # Light final smoothing around river beds only. This avoids the
        # "machine-cut trench" look while preserving nearby hills.
        river_band = water.astype(bool)
        if np.any(river_band):
            from scipy import ndimage
            terrain_range = float(np.percentile(h, 95) - np.percentile(h, 5))
            if terrain_range < 35:
                river_band = ndimage.binary_dilation(river_band, iterations=1)
                water[river_band] = 1
                core_band = ndimage.binary_dilation(river_band, iterations=2)
                blurred = _separable_gaussian(h, sigma=0.9)
                h[core_band] = blurred[core_band]

                bank_band = ndimage.binary_dilation(river_band, iterations=4)
                bank_blur = _separable_gaussian(h, sigma=0.6)
                h[bank_band] = bank_blur[bank_band]
            else:
                expanded = ndimage.binary_dilation(river_band, iterations=3 if terrain_range < 80 else 2)
                blurred = _separable_gaussian(h, sigma=1.0)
                h[expanded] = blurred[expanded]

    return water, h


# ══════════════════════════════════════════════════════════════════════════════
# MATERIALMAP
# ══════════════════════════════════════════════════════════════════════════════

def build_materialmap(
    height: np.ndarray,
    water: np.ndarray,
    flags: PromptFlags,
    sea_level: int,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Classify each cell into one of the 9 material IDs.

    Decision tree (first match wins inside each "biome" bucket):
      1. If water at this cell → water_floor (unless the biome says sand).
      2. Per-biome surface pick, biased by slope and elevation.
      3. Fallback: grass at low elevation, stone high up.
    """
    depth, width = height.shape
    mat = np.full((depth, width), NAME_TO_MATERIAL["grass"], dtype=np.uint8)

    slope = _slope_magnitude(height)
    steep = slope > 4.0
    very_steep = slope > 8.0
    high = height > (sea_level + 90)
    very_high = height > (sea_level + 150)
    low = height < (sea_level + 4)
    underwater = water == 1

    GRASS = NAME_TO_MATERIAL["grass"]
    DIRT  = NAME_TO_MATERIAL["dirt"]
    STONE = NAME_TO_MATERIAL["stone"]
    SAND  = NAME_TO_MATERIAL["sand"]
    SNOW  = NAME_TO_MATERIAL["snow"]
    ROCK  = NAME_TO_MATERIAL["rock"]
    MUD   = NAME_TO_MATERIAL["mud"]
    GRAVEL = NAME_TO_MATERIAL["gravel"]
    WATER_FLOOR = NAME_TO_MATERIAL["water_floor"]

    shore_mask = None
    river_mask = None
    lake_mask = None
    playable_feature_mask = None

    if feature_masks is not None:
        shore_mask = feature_masks.get("shore")
        river_mask = feature_masks.get("river")
        lake_mask = feature_masks.get("lake")
        playable_feature_mask = feature_masks.get("playable")

        if shore_mask is not None and shore_mask.shape != height.shape:
            shore_mask = None
        if river_mask is not None and river_mask.shape != height.shape:
            river_mask = None
        if lake_mask is not None and lake_mask.shape != height.shape:
            lake_mask = None
        if playable_feature_mask is not None and playable_feature_mask.shape != height.shape:
            playable_feature_mask = None

    # ── Baseline per-biome surface ───────────────────────────────────────────
    if flags.desert:
        mat[:] = SAND
        mat[very_steep] = ROCK
        mat[high & steep] = STONE
    elif flags.snow:
        mat[:] = GRASS
        # Snow on the upper half of the elevation range, and always on peaks.
        snow_mask = height > (sea_level + 40)
        mat[snow_mask] = SNOW
        mat[very_steep] = ROCK
    elif flags.mountain:
        mat[:] = GRASS
        mat[steep] = STONE
        mat[very_steep] = ROCK
        mat[high] = ROCK
        mat[very_high] = STONE
    elif flags.swamp:
        mat[:] = MUD
        mat[height > (sea_level + 6)] = GRASS
    elif flags.beach:
        mat[:] = GRASS
        # Everything within ~6 Y of sea level is sand. Real beach.
        beach_band = np.abs(height - sea_level) <= 6
        mat[beach_band] = SAND
    elif flags.forest or flags.plains:
        mat[:] = GRASS
        mat[very_steep] = STONE
    else:
        # No strong biome — pick based on slope/elevation only.
        mat[:] = GRASS
        mat[steep] = DIRT
        mat[very_steep] = STONE
        mat[very_high] = ROCK

    # ── Universal overrides ──────────────────────────────────────────────────
    # 1) Any cell marked as water → water_floor (unless it's a beach cell
    #    right at the waterline; there we prefer SAND so Java can put wet sand
    #    next to water cleanly).
    mat[underwater] = WATER_FLOOR
    if flags.beach:
        waterline = underwater & (np.abs(height - sea_level) <= 2)
        # Most underwater is water_floor; leave waterline as WATER_FLOOR still
        # — Java decides shoreline vs shallow-water blocks. We *don't* want
        # sand under water because that's Java's job to pick.
        del waterline  # noop, just documenting the decision

    # 2) Gravel on very-steep-but-not-quite-cliff slopes — gives Java a hint
    #    for scree material. Only apply where we haven't already placed rock.
    scree = steep & ~very_steep & ~underwater
    mat[scree & (mat == GRASS)] = GRAVEL

    # ── Feature-mask overrides ─────────────────────────────────────────────
    # These make generated scenes look intentional rather than only keyword-based.
    if feature_masks is not None:
        if lake_mask is not None:
            lake_bool = lake_mask.astype(bool)
            if np.any(lake_bool):
                mat[lake_bool] = WATER_FLOOR

        if river_mask is not None:
            river_bool = river_mask.astype(bool)
            if np.any(river_bool):
                if flags.swamp or flags.forest or flags.plains:
                    mat[river_bool] = MUD
                else:
                    mat[river_bool] = GRAVEL

        if shore_mask is not None:
            shore_bool = shore_mask.astype(bool)
            if np.any(shore_bool):
                if flags.beach or flags.desert or flags.tropical:
                    shore_mat = SAND
                elif flags.snow or flags.mountain:
                    shore_mat = GRAVEL
                elif flags.swamp or flags.forest or flags.plains:
                    shore_mat = MUD
                else:
                    shore_mat = DIRT

                dry_shore = shore_bool & (water == 0)
                mat[dry_shore] = shore_mat

        if playable_feature_mask is not None:
            playable_bool = playable_feature_mask.astype(bool)
            if np.any(playable_bool):
                dry_playable = playable_bool & (water == 0)
                if flags.desert:
                    mat[dry_playable] = SAND
                elif flags.snow:
                    if np.any(dry_playable) and float(np.mean(height[dry_playable])) > sea_level + 45:
                        mat[dry_playable] = DIRT
                    else:
                        mat[dry_playable] = GRASS
                else:
                    mat[dry_playable] = GRASS

    # 3) Clean up: ensure every cell is a valid ID.
    mat = np.clip(mat, 0, 8).astype(np.uint8)
    return mat


# ══════════════════════════════════════════════════════════════════════════════
# DECORATIONS  —  high-level placement intents only
# ══════════════════════════════════════════════════════════════════════════════

def build_decorations(
    height: np.ndarray,
    material: np.ndarray,
    water: np.ndarray,
    flags: PromptFlags,
    sea_level: int,
    origin_x: int,
    origin_z: int,
    seed: Optional[int] = None,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, List[Dict]]:
    """Pick decoration candidate positions. Java picks prefabs + validates.

    Output shape:
        {"trees": [{x, z, type, rotation, scale}, ...],
         "rocks": [{x, z, type}, ...],
         "plants": [{x, z, type}, ...]}

    Coordinates are in *world* space (origin_x/_z added to the local pixel).
    """
    rng = random.Random(seed if seed is not None else 0xDECAFBAD)
    depth, width = height.shape

    protected_mask = np.zeros((depth, width), dtype=bool)
    playable_feature_mask = np.zeros((depth, width), dtype=bool)
    shore_feature_mask = np.zeros((depth, width), dtype=bool)
    river_feature_mask = np.zeros((depth, width), dtype=bool)

    if feature_masks is not None:
        protected = feature_masks.get("protected")
        playable = feature_masks.get("playable")
        shore = feature_masks.get("shore")
        river = feature_masks.get("river")

        if protected is not None and protected.shape == height.shape:
            protected_mask = protected.astype(bool)
        if playable is not None and playable.shape == height.shape:
            playable_feature_mask = playable.astype(bool)
        if shore is not None and shore.shape == height.shape:
            shore_feature_mask = shore.astype(bool)
        if river is not None and river.shape == height.shape:
            river_feature_mask = river.astype(bool)

    slope = _slope_magnitude(height)
    GRASS = NAME_TO_MATERIAL["grass"]
    SAND  = NAME_TO_MATERIAL["sand"]
    SNOW  = NAME_TO_MATERIAL["snow"]
    MUD   = NAME_TO_MATERIAL["mud"]

    # ── Tree density + allowed types per biome ───────────────────────────────
    tree_type: Optional[str]
    if flags.forest and flags.tropical:
        tree_density = 0.008
        tree_type = "jungle"
    elif flags.forest:
        tree_density = 0.010
        tree_type = "oak"
    elif flags.snow:
        tree_density = 0.0015
        tree_type = "spruce"
    elif flags.swamp:
        tree_density = 0.003
        tree_type = "swamp_oak"
    elif flags.desert:
        # Desert gets cactus/dead bushes under "plants", no trees here.
        tree_density = 0.0
        tree_type = None
    elif flags.beach and flags.tropical:
        tree_density = 0.001
        tree_type = "palm"
    elif flags.beach:
        tree_density = 0.0
        tree_type = None
    elif flags.mountain:
        tree_density = 0.002
        tree_type = "pine"
    elif flags.plains:
        tree_density = 0.0008
        tree_type = "oak"
    elif flags.ocean:
        tree_density = 0.0
        tree_type = None
    else:
        tree_density = 0.003
        tree_type = "oak"

    trees: List[Dict] = []
    rocks: List[Dict] = []
    plants: List[Dict] = []

    total_cells = depth * width
    n_tree_candidates = int(total_cells * tree_density)

    def _valid_tree(r: int, c: int) -> bool:
        if water[r, c] == 1:                       # no trees on water
            return False
        if protected_mask[r, c] or playable_feature_mask[r, c]:
            return False
        if shore_feature_mask[r, c] or river_feature_mask[r, c]:
            return False
        if height[r, c] < sea_level:               # no trees below sea level
            return False
        if slope[r, c] > 5.0:                      # no trees on steep slopes
            return False
        m = material[r, c]
        if m == SNOW and tree_type != "spruce":    # snow only hosts spruce
            return False
        if flags.swamp and m not in (MUD, GRASS):
            return False
        return True

    for _ in range(n_tree_candidates):
        if tree_type is None:
            break
        r = rng.randint(0, depth - 1)
        c = rng.randint(0, width - 1)
        if not _valid_tree(r, c):
            continue
        trees.append({
            "x": int(origin_x + c),
            "z": int(origin_z + r),
            "type": tree_type,
            "rotation": rng.choice((0, 90, 180, 270)),
            "scale": round(rng.uniform(0.85, 1.2), 2),
        })

    # ── Rocks / boulders on steep rocky terrain ──────────────────────────────
    rock_density = 0.002 if flags.mountain else (0.0008 if flags.snow else 0.0003)
    n_rock_candidates = int(total_cells * rock_density)
    rock_types = ("small_granite", "medium_granite", "boulder", "small_slate")

    for _ in range(n_rock_candidates):
        r = rng.randint(0, depth - 1)
        c = rng.randint(0, width - 1)
        if water[r, c] == 1:                       # no rocks on water
            continue
        if protected_mask[r, c] or playable_feature_mask[r, c]:
            continue
        if slope[r, c] < 2.0 and not flags.mountain:
            # On flat plains, don't spam rocks. Allowed on mountains everywhere.
            continue
        rocks.append({
            "x": int(origin_x + c),
            "z": int(origin_z + r),
            "type": rng.choice(rock_types),
        })

    # ── Plants: cactus / dead bushes in desert, tall grass in plains ─────────
    if flags.desert:
        n_plant_candidates = int(total_cells * 0.0015)
        for _ in range(n_plant_candidates):
            r = rng.randint(0, depth - 1)
            c = rng.randint(0, width - 1)
            if material[r, c] != SAND or water[r, c] == 1:
                continue
            if slope[r, c] > 4.0:
                continue
            if protected_mask[r, c] or playable_feature_mask[r, c]:
                continue
            plants.append({
                "x": int(origin_x + c),
                "z": int(origin_z + r),
                "type": rng.choice(("cactus", "dead_bush")),
            })
    if flags.plains or flags.forest:
        n_plant_candidates = int(total_cells * 0.002)
        for _ in range(n_plant_candidates):
            r = rng.randint(0, depth - 1)
            c = rng.randint(0, width - 1)
            if material[r, c] != GRASS or water[r, c] == 1:
                continue
            if slope[r, c] > 3.0:
                continue
            if protected_mask[r, c] or playable_feature_mask[r, c]:
                continue
            plants.append({
                "x": int(origin_x + c),
                "z": int(origin_z + r),
                "type": rng.choice(("tall_grass", "fern", "flower")),
            })

    # Sparse wetland/riverbank plants near shores and rivers, but never in
    # protected/playable/water cells.
    if feature_masks is not None and (np.any(shore_feature_mask) or np.any(river_feature_mask)):
        wet_edge = (
            (shore_feature_mask | river_feature_mask)
            & (water == 0)
            & ~protected_mask
            & ~playable_feature_mask
        )
        candidates = np.argwhere(wet_edge).tolist()
        rng.shuffle(candidates)
        max_wet_plants = min(len(candidates), int(depth * width * 0.0015))
        for r, c in candidates[:max_wet_plants]:
            if slope[r, c] > 3.0:
                continue
            plants.append({
                "x": int(origin_x + int(c)),
                "z": int(origin_z + int(r)),
                "type": rng.choice(("reeds", "grass_tuft", "small_flower")),
            })

    return {"trees": trees, "rocks": rocks, "plants": plants}


# ══════════════════════════════════════════════════════════════════════════════
# IO  —  writing the gzipped binaries + JSON
# ══════════════════════════════════════════════════════════════════════════════

def _write_gz_bytes(path: str, data: bytes) -> None:
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(data)


def _write_heightmap_bin_gz(height: np.ndarray, path: str) -> None:
    """uint16 little-endian, shape [depth, width]."""
    h = np.clip(height, Y_CLAMP_MIN, Y_CLAMP_MAX).astype(np.uint16)
    # Force little-endian regardless of host byte order. np.uint16 already
    # matches on all platforms we target, but be explicit.
    buf = h.astype("<u2").tobytes(order="C")
    _write_gz_bytes(path, buf)


def _write_waterheight_bin_gz(water_height: np.ndarray, path: str) -> None:
    wh = np.clip(water_height, 0, Y_CLAMP_MAX).astype(np.uint16)
    buf = wh.astype("<u2").tobytes(order="C")
    _write_gz_bytes(path, buf)


def _write_uint8_map_gz(arr: np.ndarray, path: str) -> None:
    """uint8 map, shape [depth, width]."""
    buf = arr.astype(np.uint8).tobytes(order="C")
    _write_gz_bytes(path, buf)


def _write_decorations_json_gz(decorations: Dict, path: str) -> None:
    raw = json.dumps(decorations, separators=(",", ":")).encode("utf-8")
    _write_gz_bytes(path, raw)


def _write_preview_png(height: np.ndarray, path: str) -> None:
    """Tiny matplotlib render used purely as a sanity-check thumbnail.

    We purposely do NOT import from terrain_dataloader here — that would drag
    in torch/transformers, and the server startup already does that once; we
    don't want the package exporter to do it again on every call. The
    display range is just a visual hint, so hard-coding it is fine.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        # matplotlib optional — don't fail the export over a preview.
        return
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    vmin, vmax = -3.0, 280.0  # same as terrain_dataloader.HEIGHT_GLOBAL_{MIN,MAX}
    ax.imshow(height, cmap="terrain", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)


def build_waterheight_map(
    height: np.ndarray,
    water: np.ndarray,
    sea_level: int,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Build per-cell water surface heights."""
    h = height.astype(np.float32, copy=False)
    wh = np.zeros(h.shape, dtype=np.float32)

    water_bool = water.astype(bool)
    wh[water_bool] = float(sea_level)

    if feature_masks is not None:
        river_mask = feature_masks.get("river")
        if isinstance(river_mask, np.ndarray) and river_mask.shape == h.shape:
            river_bool = river_mask.astype(bool) & water_bool
            if np.any(river_bool):
                local_surface = h[river_bool] + 1.0
                local_surface = np.minimum(local_surface, h[river_bool] + 2.0)
                local_surface = np.maximum(local_surface, h[river_bool] + 1.0)
                wh[river_bool] = local_surface

        lake_mask = feature_masks.get("lake")
        if isinstance(lake_mask, np.ndarray) and lake_mask.shape == h.shape:
            lake_bool = lake_mask.astype(bool) & water_bool
            if np.any(lake_bool):
                wh[lake_bool] = float(sea_level)

    wh[~water_bool] = 0.0
    return np.clip(wh, 0, Y_CLAMP_MAX).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def build_terrain_package(
    stitched_raw: np.ndarray,
    prompt: Optional[str],
    directional: Dict[str, Optional[str]],
    output_dir: str,
    origin_x: int = 0,
    origin_z: int = 0,
    base_y: int = 0,
    smooth: bool = True,
    add_materials: bool = True,
    add_decorations: bool = True,
    sea_level: int = DEFAULT_SEA_LEVEL,
    smooth_strength: float = 0.3,
    feature_smoothing: bool = True,
    seed: Optional[int] = None,
    playable_mask: Optional[np.ndarray] = None,
    feature_masks: Optional[Dict[str, np.ndarray]] = None,
) -> dict:
    """Export a Java-facing terrain package. See module docstring for layout.

    Returns the parsed metadata dict (also written to disk as metadata.json).
    """
    if stitched_raw.ndim != 2:
        raise ValueError(
            f"stitched_raw must be 2D [depth, width]; got shape {stitched_raw.shape}"
        )
    depth, width = stitched_raw.shape

    flags = classify_prompt(prompt, directional)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Smooth the heightmap
    smoothed = smooth_heightmap(
        stitched_raw,
        flags=flags,
        smooth=smooth,
        strength=smooth_strength,
        feature_smoothing=feature_smoothing,
    )

    # 2. Watermap (may further modify height via river carving)
    water, height_final = build_watermap(
        smoothed, flags=flags, sea_level=sea_level, seed=seed,
    )
    island_protection = _identify_lake_islands(height_final, water, sea_level)
    if feature_masks is not None:
        water_hint = feature_masks.get("water_hint")
        river_hint = feature_masks.get("river")
        lake_hint = feature_masks.get("lake")
        playable_hint = feature_masks.get("playable")
        island_bool = island_protection & (height_final > float(sea_level) + 1.5)

        if water_hint is not None and water_hint.shape == water.shape:
            # Feature masks are a stronger signal than prompt-only water guess.
            water_candidate = water_hint.astype(bool) & ~island_bool
            water = np.maximum(water, water_candidate.astype(np.uint8))

        if lake_hint is not None and lake_hint.shape == water.shape:
            lake_bool = lake_hint.astype(bool) & ~island_bool
            if np.any(lake_bool):
                water[lake_bool] = 1
                height_final[lake_bool] = np.minimum(
                    height_final[lake_bool],
                    float(sea_level) - 2.0,
                )

        if river_hint is not None and river_hint.shape == water.shape:
            river_bool = river_hint.astype(bool)
            if np.any(river_bool):
                water[river_bool] = 1
                # Keep river bed slightly below nominal water level.
                height_final[river_bool] = np.minimum(
                    height_final[river_bool],
                    float(sea_level) - 2.0,
                )

        if playable_hint is not None and playable_hint.shape == water.shape:
            playable_bool = playable_hint.astype(bool)
            water[playable_bool] = 0
            height_final[playable_bool] = np.maximum(
                height_final[playable_bool],
                float(sea_level) + 3.0,
            )

    if playable_mask is not None:
        pm = playable_mask.astype(bool)
        if pm.shape == water.shape:
            # Playable area must stay dry.
            water[pm] = 0
            height_final[pm] = np.maximum(height_final[pm], float(sea_level) + 3.0)
    # Re-clamp after carving.
    height_final = np.clip(height_final, Y_CLAMP_MIN, Y_CLAMP_MAX)
    water_height = build_waterheight_map(
        height_final,
        water,
        sea_level,
        feature_masks=feature_masks,
    )

    # 3. Materialmap
    if add_materials:
        material = build_materialmap(
            height_final,
            water,
            flags,
            sea_level,
            feature_masks=feature_masks,
        )
    else:
        material = np.full((depth, width), NAME_TO_MATERIAL["grass"], dtype=np.uint8)

    # 4. Decorations
    if add_decorations:
        decorations = build_decorations(
            height_final,
            material,
            water,
            flags,
            sea_level,
            origin_x=origin_x,
            origin_z=origin_z,
            seed=seed,
            feature_masks=feature_masks,
        )
    else:
        decorations = {"trees": [], "rocks": [], "plants": []}

    # 5. Write files
    heightmap_path    = os.path.join(output_dir, "heightmap.bin.gz")
    materialmap_path  = os.path.join(output_dir, "materialmap.bin.gz")
    watermap_path     = os.path.join(output_dir, "watermap.bin.gz")
    waterheight_path  = os.path.join(output_dir, "waterheight.bin.gz")
    decorations_path  = os.path.join(output_dir, "decorations.json.gz")
    preview_path      = os.path.join(output_dir, "preview.png")
    metadata_path     = os.path.join(output_dir, "metadata.json")

    _write_heightmap_bin_gz(height_final, heightmap_path)
    _write_uint8_map_gz(material, materialmap_path)
    _write_uint8_map_gz(water, watermap_path)
    _write_waterheight_bin_gz(water_height, waterheight_path)
    _write_decorations_json_gz(decorations, decorations_path)
    _write_preview_png(height_final, preview_path)

    nonzero_wh = water_height[water_height > 0]

    metadata = {
        "version":          PACKAGE_VERSION,
        "width":            int(width),
        "depth":            int(depth),
        "dtype_height":     "uint16_le",
        "dtype_material":   "uint8",
        "dtype_water":      "uint8",
        "dtype_waterheight": "uint16_le",
        "originX":          int(origin_x),
        "originZ":          int(origin_z),
        "baseY":            int(base_y),
        "seaLevel":         int(sea_level),
        "seed":             None if seed is None else int(seed),
        "prompt":           prompt,
        "directionalPrompts": {k: v for k, v in (directional or {}).items()
                               if v is not None},
        "materialIds": {str(k): v for k, v in MATERIAL_IDS.items()},
        "files": {
            "heightmap":   "heightmap.bin.gz",
            "materialmap": "materialmap.bin.gz",
            "watermap":    "watermap.bin.gz",
            "waterheight": "waterheight.bin.gz",
            "decorations": "decorations.json.gz",
        },
        "counts": {
            "trees":  len(decorations["trees"]),
            "rocks":  len(decorations["rocks"]),
            "plants": len(decorations["plants"]),
        },
        "featureMaskCounts": {},
        "waterHeightStats": {
            "nonzero": int(nonzero_wh.size),
            "min": int(nonzero_wh.min()) if nonzero_wh.size else 0,
            "max": int(nonzero_wh.max()) if nonzero_wh.size else 0,
        },
    }
    if feature_masks is not None:
        metadata["featureMaskCounts"] = {
            name: int(np.sum(mask > 0))
            for name, mask in feature_masks.items()
            if isinstance(mask, np.ndarray)
        }
    metadata["materialCounts"] = {
        MATERIAL_IDS[int(mat_id)]: int(count)
        for mat_id, count in zip(*np.unique(material, return_counts=True))
    }
    metadata["heightmapSha1"] = hashlib.sha1(
        np.clip(height_final, Y_CLAMP_MIN, Y_CLAMP_MAX).astype("<u2").tobytes(order="C")
    ).hexdigest()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
