from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import ndimage

from terrain_dataloader import normalize_height


_PARSER_TO_MACRO_SHAPE = {
    "bowl": "mountain_basin",
    "canyon": "canyon",
    "coast": "coastline",
    "peak": "mountain_basin",
    "ridge": "mountain_basin",
    "plateau": "mesa_plateau",
    "dunes": "dunes",
    "rolling": "plains",
    "swamp": "swamp_basin",
    "lake_basin": "lake_basin",
    "crater": "crater",
    "island": "island",
    "river_valley": "river_valley",
    "archipelago": "archipelago",
}


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _pick_base_shape(macro_params: Dict[str, Any]) -> str:
    primitive = str(macro_params.get("primitive", "rolling")).lower().strip()
    return _PARSER_TO_MACRO_SHAPE.get(primitive, primitive)


def add_composition_variation(params: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(params)
    seed = p.get("seed")
    if seed is None:
        return p

    rng = np.random.default_rng(int(seed))
    base_shape = str(p.get("base_shape", "")).lower()

    def set_if_missing(key: str, value: Any) -> None:
        if key not in p or p.get(key) is None:
            p[key] = value

    if base_shape == "river_valley":
        set_if_missing("river_p0_y", float(rng.uniform(-0.55, 0.55)))
        set_if_missing("river_p1_x", float(rng.uniform(-0.75, -0.25)))
        set_if_missing("river_p1_y", float(rng.uniform(-0.85, 0.85)))
        set_if_missing("river_p2_x", float(rng.uniform(0.20, 0.75)))
        set_if_missing("river_p2_y", float(rng.uniform(-0.85, 0.85)))
        set_if_missing("river_p3_y", float(rng.uniform(-0.55, 0.55)))
        set_if_missing("river_angle_degrees", float(rng.uniform(-25.0, 25.0)))
        set_if_missing("river_meander_scale", float(rng.uniform(0.7, 1.5)))
        set_if_missing("valley_side_bias", float(rng.uniform(-0.22, 0.22)))
    elif base_shape in {"canyon", "bezier_canyon"}:
        set_if_missing("canyon_p0_y", float(rng.uniform(-0.60, 0.60)))
        set_if_missing("canyon_p1_x", float(rng.uniform(-0.85, -0.20)))
        set_if_missing("canyon_p1_y", float(rng.uniform(-0.95, 0.95)))
        set_if_missing("canyon_p2_x", float(rng.uniform(0.20, 0.85)))
        set_if_missing("canyon_p2_y", float(rng.uniform(-0.95, 0.95)))
        set_if_missing("canyon_p3_y", float(rng.uniform(-0.60, 0.60)))
        set_if_missing("canyon_angle_degrees", float(rng.uniform(-35.0, 35.0)))
        set_if_missing("canyon_width_scale", float(rng.uniform(0.75, 1.35)))
        set_if_missing("canyon_depth_scale", float(rng.uniform(0.85, 1.25)))
        set_if_missing("canyon_rim_bias", float(rng.uniform(-0.20, 0.20)))
    elif base_shape == "lake_basin":
        set_if_missing("lake_center_x", float(rng.uniform(-0.35, 0.35)))
        set_if_missing("lake_center_y", float(rng.uniform(-0.25, 0.30)))
        set_if_missing("lake_radius", float(rng.uniform(0.22, 0.34)))
        set_if_missing("lake_aspect_x", float(rng.uniform(0.75, 1.30)))
        set_if_missing("lake_aspect_y", float(rng.uniform(0.75, 1.30)))
        set_if_missing("lake_wobble_1", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("lake_wobble_2", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("lake_wobble_3", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("lake_outlet_angle", float(rng.uniform(-math.pi, math.pi)))
        island_roll = float(rng.uniform(0.0, 1.0))
        set_if_missing("lake_island_chance", island_roll)
        set_if_missing("lake_island_x", float(rng.uniform(-0.18, 0.18)))
        set_if_missing("lake_island_y", float(rng.uniform(-0.15, 0.15)))
        set_if_missing("lake_island_radius", float(rng.uniform(0.05, 0.14)))
        set_if_missing("lake_island_height", float(rng.uniform(5.0, 12.0)))
    elif base_shape == "mountain_basin":
        set_if_missing("basin_center_x", float(rng.uniform(-0.22, 0.22)))
        set_if_missing("basin_center_y", float(rng.uniform(-0.18, 0.18)))
        set_if_missing("basin_aspect_x", float(rng.uniform(0.80, 1.25)))
        set_if_missing("basin_aspect_y", float(rng.uniform(0.80, 1.25)))
        set_if_missing("pass_angle_1", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("pass_angle_2", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("ridge_phase_1", float(rng.uniform(-math.pi, math.pi)))
        set_if_missing("ridge_phase_2", float(rng.uniform(-math.pi, math.pi)))
    elif base_shape == "plains":
        set_if_missing("plains_trend_angle", float(rng.uniform(-35.0, 35.0)))
        set_if_missing("plains_drainage_bias", float(rng.uniform(-3.0, 3.0)))
        set_if_missing("plains_micro_seed_offset", int(rng.integers(1, 1_000_000)))

    return p


def summarize_composition(params: Dict[str, Any]) -> str:
    base_shape = str(params.get("base_shape", "")).lower()
    if base_shape == "river_valley":
        return (
            "river="
            f"({float(params.get('river_p0_y', 0.0)):.2f};"
            f"{float(params.get('river_p1_x', 0.0)):.2f},{float(params.get('river_p1_y', 0.0)):.2f};"
            f"{float(params.get('river_p2_x', 0.0)):.2f},{float(params.get('river_p2_y', 0.0)):.2f};"
            f"{float(params.get('river_p3_y', 0.0)):.2f}) "
            f"ang={float(params.get('river_angle_degrees', 0.0)):.1f} "
            f"meander={float(params.get('river_meander_scale', 1.0)):.2f}"
        )
    if base_shape in {"canyon", "bezier_canyon"}:
        return (
            "canyon="
            f"({float(params.get('canyon_p0_y', 0.0)):.2f};"
            f"{float(params.get('canyon_p1_x', 0.0)):.2f},{float(params.get('canyon_p1_y', 0.0)):.2f};"
            f"{float(params.get('canyon_p2_x', 0.0)):.2f},{float(params.get('canyon_p2_y', 0.0)):.2f};"
            f"{float(params.get('canyon_p3_y', 0.0)):.2f}) "
            f"ang={float(params.get('canyon_angle_degrees', 0.0)):.1f} "
            f"w={float(params.get('canyon_width_scale', 1.0)):.2f} "
            f"d={float(params.get('canyon_depth_scale', 1.0)):.2f}"
        )
    if base_shape == "lake_basin":
        return (
            f"lake=({float(params.get('lake_center_x', 0.0)):.2f},"
            f"{float(params.get('lake_center_y', 0.0)):.2f}) "
            f"r={float(params.get('lake_radius', 0.0)):.2f} "
            f"aspect=({float(params.get('lake_aspect_x', 1.0)):.2f},"
            f"{float(params.get('lake_aspect_y', 1.0)):.2f}) "
            f"island={float(params.get('lake_island_chance', 0.0)) > 0.55}"
        )
    return ""


def adapt_macro_params(
    macro_params: Dict[str, Any],
    *,
    base_y: float,
    sea_level: float,
) -> Dict[str, Any]:
    p = dict(macro_params)
    p["base_shape"] = _pick_base_shape(p)

    if "world_size" in p:
        p["size"] = int(p["world_size"])

    primitive_params = p.get("primitive_params")
    if isinstance(primitive_params, dict):
        p.update(primitive_params)
    else:
        primitive_params = {}

    text = str(p.get("_matched_text", "")).lower()
    primitive = str(p.get("primitive", "rolling")).lower().strip()
    rivers = p.get("rivers") or {}
    river_count = int(rivers.get("count", 0) or 0)

    if primitive == "rolling":
        # Parser uses "rolling" for plains/meadows/grasslands as well.
        # The adapter now maps this to terrain_macro primitive_plains by default.
        amp = float(primitive_params.get("amplitude_y", 12.0))

        if _has_any(text, ("plain", "plains", "grassland", "meadow", "prairie", "steppe", "savanna", "savannah")):
            p["base_shape"] = "plains"
            p["plains_relief"] = min(amp, 3.5)
            p["plains_broad_relief"] = 1.8
            p["noise_amplitude"] = min(float(p.get("noise_amplitude", 5.0)), 1.5)
            p["final_blur_sigma"] = max(float(p.get("final_blur_sigma", 1.25)), 2.2)
        else:
            # Non-plains fallback can still be gentle hills, but much less extreme
            # than terrain_macro's old hill_height default.
            p["base_shape"] = "rolling_hills"
            p["hill_height"] = min(amp, 14.0)
            p["noise_amplitude"] = min(float(p.get("noise_amplitude", 5.0)), 4.0)

    elif primitive == "lake_basin":
        modifiers = p.get("modifiers") or {}
        # True lake-specific macro. Keep the lake basin intentional instead of
        # using generic mountain_basin + later flood fill.
        lake_depth = float(primitive_params.get("depth_y", 8.0))
        lake_radius = float(primitive_params.get("lake_radius_frac", 0.22))
        surround = float(primitive_params.get("surround_height_y", 18.0))

        p["base_shape"] = "lake_basin"
        p["lake_radius"] = max(0.18, min(0.32, lake_radius))
        p["lake_depth"] = max(7.0, lake_depth)
        p["lake_shore_width"] = 0.10
        p["lake_shelf_height"] = 5.0

        if modifiers.get("snowy") or _has_any(text, ("alpine", "mountain", "snow", "tarn", "loch")):
            p["lake_surround_height"] = max(85.0, surround * 5.0)
            p["noise_amplitude"] = 5.0
            p["final_blur_sigma"] = 1.2
        else:
            p["lake_surround_height"] = max(30.0, surround * 1.8)
            p["noise_amplitude"] = 3.5
            p["final_blur_sigma"] = 1.5

    if river_count > 0:
        if _has_any(text, ("plain", "plains", "grassland", "meadow", "prairie", "steppe", "savanna", "savannah")):
            p["base_shape"] = "river_valley"
            p["valley_depth"] = 8.0
            p["valley_width"] = max(float(p.get("valley_width", 0.16)), 0.28)
            p["noise_amplitude"] = min(float(p.get("noise_amplitude", 5.0)), 1.5)
            p["final_blur_sigma"] = max(float(p.get("final_blur_sigma", 1.25)), 2.2)
        else:
            p["base_shape"] = "river_valley"

    ocean = p.get("ocean")
    if isinstance(ocean, dict):
        side = ocean.get("side")
        if side:
            p["ocean_side"] = side

    p["water_level"] = float(sea_level)
    p["base_elevation"] = float(base_y)
    p["base_y"] = float(base_y)
    return add_composition_variation(p)


def _prompt_text_from_params(params: Dict[str, Any]) -> str:
    return str(params.get("_matched_text") or params.get("_prompt") or "").lower()


def _soft_irregular_mask(
    shape: Tuple[int, int],
    center_r: float,
    center_c: float,
    radius: float,
    feather: float,
    seed: Optional[int],
) -> np.ndarray:
    """Create a soft irregular circular-ish mask in [0, 1].

    1.0 means strongly protected/playable center.
    0.0 means unaffected terrain.
    """
    depth, width = shape
    yy, xx = np.mgrid[0:depth, 0:width].astype(np.float32)

    dy = yy - float(center_r)
    dx = xx - float(center_c)
    theta = np.arctan2(dy, dx)
    dist = np.sqrt(dx * dx + dy * dy)

    # Low-frequency angular wobble so the area is not a perfect circle.
    rng = np.random.default_rng(seed)
    phase1 = float(rng.uniform(-math.pi, math.pi))
    phase2 = float(rng.uniform(-math.pi, math.pi))
    phase3 = float(rng.uniform(-math.pi, math.pi))

    wobble = (
        0.12 * np.sin(3.0 * theta + phase1)
        + 0.08 * np.sin(5.0 * theta + phase2)
        + 0.05 * np.sin(8.0 * theta + phase3)
    )
    local_radius = float(radius) * (1.0 + wobble)

    inner = np.clip((local_radius - dist) / max(float(feather), 1e-6), 0.0, 1.0)
    mask = inner * inner * (3.0 - 2.0 * inner)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _find_low_slope_candidate(
    height: np.ndarray,
    avoid_water_level: float,
    prefer_near_level: Optional[float],
    seed: Optional[int],
) -> Tuple[int, int]:
    """Pick a generally usable point from the macro heightmap.

    This is a fallback picker. It prefers low slope, dry cells, and optionally
    cells near a target level such as lake shore or river terrace.
    """
    depth, width = height.shape
    rng = np.random.default_rng(seed)

    gy, gx = np.gradient(height.astype(np.float32))
    slope = np.sqrt(gx * gx + gy * gy)

    margin = max(24, min(depth, width) // 12)
    valid = np.ones(height.shape, dtype=bool)
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False

    # Must be dry-ish.
    valid &= height > (float(avoid_water_level) + 1.5)

    if prefer_near_level is not None:
        level_score = np.abs(height - float(prefer_near_level))
    else:
        level_score = np.abs(height - np.percentile(height, 35))

    # Lower score is better.
    score = slope * 4.0 + level_score * 0.35

    # Add tiny random jitter so same prompt/seed is stable but avoids ties.
    score = score + rng.normal(0.0, 0.01, size=score.shape)

    score = np.where(valid, score, np.inf)
    if not np.isfinite(score).any():
        return depth // 2, width // 2

    idx = int(np.argmin(score))
    r, c = np.unravel_index(idx, score.shape)
    return int(r), int(c)


def choose_playable_area(
    height: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Choose a natural playable area for the scene.

    Returns a dict with:
      mask: float32 [H,W] in [0,1]
      center_r, center_c
      radius
      feather
      target_height
    """
    text = _prompt_text_from_params(adapted_params)
    base_shape = str(adapted_params.get("base_shape", "")).lower()

    depth, width = height.shape
    rng = np.random.default_rng(seed)

    # Roughly 32x32 usable area. Radius 18 with feather gives a natural blob.
    radius = float(adapted_params.get("playable_radius", 20.0))
    feather = float(adapted_params.get("playable_feather", 14.0))

    center_r: int
    center_c: int
    prefer_level: Optional[float] = None

    # Lake scenes: pick a shore shelf, not lake center.
    if "lake" in base_shape or "lake" in text or "tarn" in text or "loch" in text:
        lake_cx = float(adapted_params.get("lake_center_x", -0.12))
        lake_cy = float(adapted_params.get("lake_center_y", 0.08))
        lake_radius = float(adapted_params.get("lake_radius", 0.28))
        shore_width = float(adapted_params.get("lake_shore_width", 0.08))

        # Convert normalized terrain coords [-1,1] to pixel coords.
        lake_center_c = (lake_cx + 1.0) * 0.5 * (width - 1)
        lake_center_r = (lake_cy + 1.0) * 0.5 * (depth - 1)

        angle = float(adapted_params.get("playable_lake_angle", 2.35))
        dist_norm = lake_radius + shore_width * 1.7
        center_c = int(np.clip(lake_center_c + math.cos(angle) * dist_norm * 0.5 * width, 24, width - 25))
        center_r = int(np.clip(lake_center_r + math.sin(angle) * dist_norm * 0.5 * depth, 24, depth - 25))
        prefer_level = float(sea_level) + 4.0

        # If chosen shore is too wet/steep, fallback to lowest dry shelf.
        if height[center_r, center_c] <= sea_level + 1:
            center_r, center_c = _find_low_slope_candidate(
                height,
                avoid_water_level=sea_level,
                prefer_near_level=float(sea_level) + 5.0,
                seed=seed,
            )

    # River scenes: choose a dry terrace near the central river corridor.
    elif "river" in base_shape or "river" in text or "stream" in text or "creek" in text:
        # Current river_valley path roughly crosses the center. Pick an offset
        # terrace near the middle, not directly on the channel.
        center_r = int(np.clip(depth * 0.52 + rng.uniform(-depth * 0.08, depth * 0.08), 24, depth - 25))
        center_c = int(np.clip(width * 0.48 + rng.uniform(-width * 0.08, width * 0.08), 24, width - 25))

        if height[center_r, center_c] <= sea_level + 1:
            center_r, center_c = _find_low_slope_candidate(
                height,
                avoid_water_level=sea_level,
                prefer_near_level=float(sea_level) + 8.0,
                seed=seed,
            )
        prefer_level = float(height[center_r, center_c])

    # Canyon scenes: use a widened floor/shelf candidate.
    elif "canyon" in base_shape or "canyon" in text or "gorge" in text:
        center_r, center_c = _find_low_slope_candidate(
            height,
            avoid_water_level=sea_level,
            prefer_near_level=float(np.percentile(height, 25)),
            seed=seed,
        )
        prefer_level = float(height[center_r, center_c])

    # Default: low-slope dry area.
    else:
        center_r, center_c = _find_low_slope_candidate(
            height,
            avoid_water_level=sea_level,
            prefer_near_level=None,
            seed=seed,
        )
        prefer_level = float(height[center_r, center_c])

    mask = _soft_irregular_mask(
        height.shape,
        center_r=center_r,
        center_c=center_c,
        radius=radius,
        feather=feather,
        seed=seed,
    )

    # Target height is robust local median, not a hard global value.
    strong = mask > 0.75
    medium = mask > 0.25

    if np.any(strong):
        target = float(np.median(height[strong]))
    elif prefer_level is not None:
        target = float(prefer_level)
    else:
        target = float(height[center_r, center_c])

    # Ensure dry area.
    target = max(target, float(sea_level) + 3.0)

    return {
        "mask": mask.astype(np.float32),
        "center_r": int(center_r),
        "center_c": int(center_c),
        "radius": float(radius),
        "feather": float(feather),
        "target_height": float(target),
        "medium_mask": medium.astype(np.uint8),
    }


def apply_playable_area(
    height: np.ndarray,
    playable: Dict[str, Any],
    *,
    sea_level: int = 63,
    strength: float = 0.92,
) -> np.ndarray:
    """Blend an irregular playable area into a heightmap.

    The center becomes gently usable, while feathered edges blend into the
    surrounding terrain. This avoids a hard square arena.
    """
    h = height.astype(np.float32, copy=True)
    mask = playable["mask"].astype(np.float32)
    target = float(playable["target_height"])

    # Smooth local terrain to remove bumps inside playable zone.
    local_smooth = ndimage.gaussian_filter(h, sigma=3.0, mode="reflect")

    strong_mask = mask > 0.5
    if np.any(strong_mask):
        local_mean = float(np.mean(local_smooth[strong_mask]))
    else:
        local_mean = float(np.mean(local_smooth))

    # Flatten only the strong center, but retain slight natural variation.
    usable = target + (local_smooth - local_mean) * 0.12
    usable = np.maximum(usable, float(sea_level) + 3.0)

    blend = np.clip(mask * float(strength), 0.0, 1.0)
    out = h * (1.0 - blend) + usable * blend
    if np.any(strong_mask):
        out[strong_mask] = np.maximum(out[strong_mask], float(sea_level) + 3.0)

    return out.astype(np.float32)


def build_constraint_masks(
    height: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
) -> Dict[str, np.ndarray]:
    """Build soft masks for areas that should survive img2img refinement.

    These masks are in macro/world pixel space and should match the final
    stitched heightmap shape.

    Returns float32 masks in [0, 1]:
      playable: from _playable_mask if present
      water: likely lake/water basin areas
      shore: near-water transition band
      river: approximate river corridor for river_valley scenes
      protect: combined mask
    """
    h = height.astype(np.float32, copy=False)
    depth, width = h.shape

    zero = np.zeros((depth, width), dtype=np.float32)

    playable = adapted_params.get("_playable_mask")
    if playable is None:
        playable_mask = zero.copy()
    else:
        playable_mask = np.asarray(playable, dtype=np.float32)
        if playable_mask.shape != h.shape:
            playable_mask = zero.copy()
        else:
            playable_mask = np.clip(playable_mask, 0.0, 1.0)

    base_shape = str(adapted_params.get("base_shape", "")).lower()
    text = _prompt_text_from_params(adapted_params)

    # Water-like mask from macro height. Keep this conservative.
    water_mask = zero.copy()
    shore_mask = zero.copy()

    if (
        "lake" in base_shape
        or "lake" in text
        or "tarn" in text
        or "loch" in text
        or "ocean" in text
        or "coast" in text
        or "shore" in text
    ):
        # Water is the low basin around/under sea level.
        likely_water = h < (float(sea_level) + 0.75)
        if np.any(likely_water):
            labels, count = ndimage.label(likely_water)
            if count > 0:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                largest = int(np.argmax(sizes))
                water_bool = labels == largest
            else:
                water_bool = likely_water

            water_bool = ndimage.binary_dilation(water_bool, iterations=1)
            water_mask = water_bool.astype(np.float32)

            shore_bool = ndimage.binary_dilation(water_bool, iterations=8) & ~ndimage.binary_erosion(water_bool, iterations=1)
            shore_mask = shore_bool.astype(np.float32)
            shore_mask = ndimage.gaussian_filter(shore_mask, sigma=2.0, mode="reflect")
            shore_mask = np.clip(shore_mask, 0.0, 1.0)

    # Approximate river mask. This protects the macro river valley/corridor
    # before the exporter adds final watermap details.
    river_mask = zero.copy()
    if "river" in base_shape or "river" in text or "stream" in text or "creek" in text:
        # River corridor in terrain_macro.primitive_river_valley uses a fixed
        # Bezier-like path from left to right. Recreate a broad approximate
        # corridor in pixel space.
        yy, xx = np.mgrid[0:depth, 0:width].astype(np.float32)
        xn = (xx / max(width - 1, 1)) * 2.0 - 1.0
        yn = (yy / max(depth - 1, 1)) * 2.0 - 1.0

        angle = math.radians(float(adapted_params.get("river_angle_degrees", 0.0)))
        ca, sa = math.cos(angle), math.sin(angle)

        def rot(px: float, py: float) -> np.ndarray:
            return np.array([px * ca - py * sa, px * sa + py * ca], dtype=np.float32)

        p0 = rot(-1.05, float(adapted_params.get("river_p0_y", -0.10)))
        p1 = rot(
            float(adapted_params.get("river_p1_x", -0.40)),
            float(adapted_params.get("river_p1_y", -0.55)),
        )
        p2 = rot(
            float(adapted_params.get("river_p2_x", 0.15)),
            float(adapted_params.get("river_p2_y", 0.45)),
        )
        p3 = rot(1.05, float(adapted_params.get("river_p3_y", 0.05)))

        samples = 256
        ts = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        path = []
        for t in ts:
            u = 1.0 - t
            pt = (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3
            path.append(pt)
        path = np.asarray(path, dtype=np.float32)

        # Distance to sampled path. This is O(samples * pixels), okay for current sizes.
        dist2 = np.full((depth, width), np.inf, dtype=np.float32)
        for px, py in path:
            d2 = (xn - px) ** 2 + (yn - py) ** 2
            dist2 = np.minimum(dist2, d2)

        dist = np.sqrt(dist2)
        corridor_width = float(adapted_params.get("valley_width", 0.20))
        river_mask = np.clip(1.0 - dist / max(corridor_width * 1.3, 1e-6), 0.0, 1.0)
        river_mask = river_mask.astype(np.float32)
        river_mask = ndimage.gaussian_filter(river_mask, sigma=1.5, mode="reflect")
        river_mask = np.clip(river_mask, 0.0, 1.0)

    protect = np.maximum.reduce([
        playable_mask * 1.0,
        water_mask * 0.65,
        shore_mask * 0.25,
        river_mask * 0.30,
    ]).astype(np.float32)

    return {
        "playable": playable_mask.astype(np.float32),
        "water": water_mask.astype(np.float32),
        "shore": shore_mask.astype(np.float32),
        "river": river_mask.astype(np.float32),
        "protect": protect,
    }


def build_feature_masks(
    height: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
) -> Dict[str, np.ndarray]:
    """Build explicit feature masks for export.

    This is a lightweight FeatureMap bridge. It does not change the exporter
    file format yet. It gives terrain_package.py better information than
    prompt keyword re-parsing alone.

    Returned masks are uint8 [H,W]:
      water_hint: likely water cells from macro scene
      lake: lake basin water cells
      shore: lake/ocean shore band
      river: river corridor / wet channel hint
      playable: protected playable area
      protected: union of important functional areas
    """
    h = height.astype(np.float32, copy=False)
    depth, width = h.shape

    masks = build_constraint_masks(h, adapted_params, sea_level=sea_level)

    playable = masks.get("playable", np.zeros_like(h)) > 0.25
    shore = masks.get("shore", np.zeros_like(h)) > 0.20
    river_soft = masks.get("river", np.zeros_like(h))
    river = river_soft > 0.72

    base_shape = str(adapted_params.get("base_shape", "")).lower()
    text = _prompt_text_from_params(adapted_params)

    lake = np.zeros_like(h, dtype=bool)
    water_hint = np.zeros_like(h, dtype=bool)

    if "lake" in base_shape or "lake" in text or "tarn" in text or "loch" in text:
        # Conservative lake water: largest connected low basin near water level.
        low = h < (float(sea_level) + 0.25)
        if np.any(low):
            labels, count = ndimage.label(low)
            if count > 0:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                largest = int(np.argmax(sizes))
                lake = labels == largest
                lake = ndimage.binary_dilation(lake, iterations=1)
                lake &= h < (float(sea_level) + 1.5)

    # River water hint: narrow part of the protected river corridor.
    if "river" in base_shape or "river" in text or "stream" in text or "creek" in text:
        river = river_soft > 0.78

    water_hint |= lake
    water_hint |= river

    # Oceans/coasts can still be broad sea-level water.
    if "coast" in base_shape or "ocean" in text or "sea" in text:
        water_hint |= h < float(sea_level)

    # Never mark playable as water.
    water_hint &= ~playable
    lake &= ~playable
    river &= ~playable

    protected = playable | shore | river | lake

    return {
        "water_hint": water_hint.astype(np.uint8),
        "lake": lake.astype(np.uint8),
        "shore": shore.astype(np.uint8),
        "river": river.astype(np.uint8),
        "playable": playable.astype(np.uint8),
        "protected": protected.astype(np.uint8),
    }


def apply_constraint_repair(
    ai_height: np.ndarray,
    macro_height: np.ndarray,
    adapted_params: Dict[str, Any],
    *,
    sea_level: int = 63,
    strength: float = 0.70,
) -> np.ndarray:
    """Blend important macro constraints back into img2img output.

    ai_height:
      raw denormalized output after img2img.

    macro_height:
      raw macro height used before normalization and img2img. This should
      include playable-area blending from Phase 5.

    adapted_params:
      adapted macro params containing _playable_mask and scene data.

    This is not a full FeatureMap yet. It is a compatibility step to keep
    playable/water/river structure stable until FeatureMap cleanup.
    """
    ai = ai_height.astype(np.float32, copy=True)
    macro = macro_height.astype(np.float32, copy=False)

    if ai.shape != macro.shape:
        return ai

    masks = build_constraint_masks(macro, adapted_params, sea_level=sea_level)
    protect = np.clip(masks["protect"] * float(strength), 0.0, 1.0)

    if np.max(protect) <= 0.0:
        return ai

    # Smooth macro only lightly so protected areas still have natural feathering.
    macro_smooth = ndimage.gaussian_filter(macro, sigma=0.9, mode="reflect")

    repaired = ai * (1.0 - protect) + macro_smooth * protect

    # Strongly preserve playable center as usable.
    playable = masks["playable"]
    if np.max(playable) > 0.0:
        playable_meta = adapted_params.get("_playable_area") or {}
        playable_obj = dict(playable_meta)
        playable_obj["mask"] = playable

        repaired = apply_playable_area(
            repaired,
            playable_obj,
            sea_level=sea_level,
            strength=0.82,
        )

    # Keep likely lake/water basin below or near water level, but avoid
    # flattening everything. The actual watermap is still built later.
    water = masks["water"] > 0.5
    if np.any(water):
        repaired[water] = np.minimum(repaired[water], float(sea_level) - 1.0)

    # Keep shore dry-ish and not extremely jagged.
    shore = masks["shore"] > 0.15
    if np.any(shore):
        shore_smooth = ndimage.gaussian_filter(repaired, sigma=1.4, mode="reflect")
        repaired[shore] = repaired[shore] * 0.65 + shore_smooth[shore] * 0.35
        repaired[shore] = np.maximum(repaired[shore], float(sea_level) + 1.0)

    river = masks["river"] > 0.2
    if np.any(river):
        river_smooth = ndimage.gaussian_filter(repaired, sigma=1.0, mode="reflect")
        repaired[river] = repaired[river] * 0.8 + river_smooth[river] * 0.2

    return repaired.astype(np.float32)


def build_normalized_macro_world(
    terrain_macro_module,
    parser_params: Dict[str, Any],
    *,
    base_y: int = 63,
    sea_level: int = 63,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
    """
    Build terrain_macro raw heightmap, add a natural playable area, and
    normalize it for img2img.

    Returns:
      macro_world_norm, adapted_params, debug_stats

    Also stores playable metadata in adapted_params["_playable_area"].
    """
    adapted = adapt_macro_params(
        parser_params,
        base_y=base_y,
        sea_level=sea_level,
    )

    macro_raw = terrain_macro_module.generate_macro_shape(adapted).astype(np.float32)

    playable = choose_playable_area(
        macro_raw,
        adapted,
        sea_level=sea_level,
        seed=adapted.get("seed"),
    )

    macro_playable = apply_playable_area(
        macro_raw,
        playable,
        sea_level=sea_level,
        strength=float(adapted.get("playable_strength", 0.92)),
    )

    adapted["_playable_area"] = {
        "center_r": playable["center_r"],
        "center_c": playable["center_c"],
        "radius": playable["radius"],
        "feather": playable["feather"],
        "target_height": playable["target_height"],
    }

    # Keep the actual mask available for current-process post-img2img repair.
    adapted["_playable_mask"] = playable["mask"]
    adapted["_macro_raw_with_playable"] = macro_playable.astype(np.float32)

    constraint_masks = build_constraint_masks(
        macro_playable,
        adapted,
        sea_level=sea_level,
    )
    adapted["_constraint_masks"] = constraint_masks
    feature_masks = build_feature_masks(
        macro_playable,
        adapted,
        sea_level=sea_level,
    )
    adapted["_feature_masks"] = feature_masks

    stats = {
        "raw_min": float(np.min(macro_raw)),
        "raw_max": float(np.max(macro_raw)),
        "raw_mean": float(np.mean(macro_raw)),
        "playable_min": float(np.min(macro_playable)),
        "playable_max": float(np.max(macro_playable)),
        "playable_mean": float(np.mean(macro_playable)),
        "playable_center_r": float(playable["center_r"]),
        "playable_center_c": float(playable["center_c"]),
        "playable_target": float(playable["target_height"]),
    }
    stats["protect_max"] = float(np.max(constraint_masks["protect"]))
    stats["water_mask_cells"] = float(np.sum(constraint_masks["water"] > 0.5))
    stats["river_mask_cells"] = float(np.sum(constraint_masks["river"] > 0.5))
    stats["feature_water_cells"] = float(np.sum(feature_masks["water_hint"] > 0))
    stats["feature_playable_cells"] = float(np.sum(feature_masks["playable"] > 0))
    stats["feature_lake_cells"] = float(np.sum(feature_masks["lake"] > 0))
    stats["feature_river_cells"] = float(np.sum(feature_masks["river"] > 0))

    macro_norm = normalize_height(macro_playable).astype(np.float32)

    stats.update({
        "norm_min": float(np.min(macro_norm)),
        "norm_max": float(np.max(macro_norm)),
        "norm_mean": float(np.mean(macro_norm)),
    })

    return macro_norm, adapted, stats
