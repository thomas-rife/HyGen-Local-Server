"""
terrain_macro.py

Universal Procedural Macro-Shape Generator for terrain heightmaps.

Purpose
-------
This module builds the large-scale terrain shape BEFORE AI/diffusion/detail passes.

It is designed to replace a hardcoded flat center chunk with a prompt-aware
global terrain guide, for example:

    - mountain basin surrounded by rims
    - canyon floor with high walls
    - coastline / beach slope
    - rolling hills
    - dunes
    - crater
    - island
    - river valley
    - mesa / plateau
    - swamp basin
    - archipelago

No diffusion code is included here.

Dependencies
------------
Required:
    numpy
    scipy

Optional:
    numba

Main entrypoints
----------------
    generate_macro_shape(params) -> np.ndarray
    hydraulic_erosion(height, erosion_params=None) -> np.ndarray

Example
-------
    params = {
        "base_shape": "mountain_basin",
        "size": 1024,
        "base_elevation": 64,
        "center_depth": 22,
        "rim_height": 120,
        "noise_amplitude": 8,
        "seed": 123,
    }

    h = generate_macro_shape(params)
    h = hydraulic_erosion(h, {"iterations": 40000, "seed": 123})
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco


Array = np.ndarray
ParamsLike = Union[str, Dict[str, Any]]


# =============================================================================
# PARAMETER HANDLING
# =============================================================================

DEFAULT_PARAMS: Dict[str, Any] = {
    "base_shape": "rolling_hills",
    "size": 1024,
    "base_elevation": 64.0,
    "min_height": 0.0,
    "max_height": 255.0,

    # Common shaping controls
    "center_depth": 20.0,
    "rim_height": 90.0,
    "radius": 0.55,
    "inner_radius": 0.18,
    "outer_radius": 0.85,
    "wall_steepness": 2.5,
    "slope": 60.0,

    # Noise/detail controls
    "noise_amplitude": 8.0,
    "noise_octaves": 5,
    "noise_persistence": 0.5,
    "noise_lacunarity": 2.0,
    "noise_blur_sigma": 8.0,

    # Orientation and variation
    "angle_degrees": 0.0,
    "seed": None,

    # Feature controls
    "water_level": 63.0,
    "canyon_width": 0.075,
    "canyon_depth": 85.0,
    "dune_height": 22.0,
    "dune_frequency": 11.0,
    "hill_height": 38.0,
    "plains_relief": 3.5,
    "plains_broad_relief": 1.8,
    "crater_depth": 70.0,
    "crater_rim_height": 45.0,
    "island_height": 90.0,
    "coast_position": 0.0,
    "coast_amplitude": 0.18,
    "coast_frequency": 2.0,
    "lake_radius": 0.26,
    "lake_depth": 8.0,
    "lake_surround_height": 75.0,
    "lake_shore_width": 0.10,
    "lake_shelf_height": 5.0,
    "lake_center_x": -0.12,
    "lake_center_y": 0.08,
    "lake_outlet_angle": -0.65,
    "valley_width": 0.16,
    "valley_depth": 45.0,
}


def _coerce_params(params: ParamsLike) -> Dict[str, Any]:
    if isinstance(params, str):
        user = json.loads(params)
    else:
        user = dict(params)

    out = dict(DEFAULT_PARAMS)
    out.update(user)
    return out


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _clamp_height(h: Array, params: Dict[str, Any]) -> Array:
    return np.clip(
        h.astype(np.float32, copy=False),
        float(params["min_height"]),
        float(params["max_height"]),
    )


def _coords(size: int) -> Tuple[Array, Array, Array]:
    """
    Returns x, y, radial distance arrays.

    x and y are in [-1, 1].
    y increases downward in image/map coordinates.
    """
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x * x + y * y)
    return x, y, r


def _rotate(x: Array, y: Array, angle_degrees: float) -> Tuple[Array, Array]:
    a = math.radians(angle_degrees)
    ca, sa = math.cos(a), math.sin(a)
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr, yr


def smoothstep(edge0: float, edge1: float, x: Array) -> Array:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normalize01(a: Array) -> Array:
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


# =============================================================================
# LOW-FREQUENCY NOISE
# =============================================================================

def value_noise(
    size: int,
    grid: int,
    rng: np.random.Generator,
    blur_sigma: float = 0.0,
) -> Array:
    """
    Smooth value noise made from a coarse random grid upsampled to size.

    This is intentionally low-frequency and controllable, not high-detail terrain.
    """
    grid = max(2, int(grid))
    coarse = rng.normal(0.0, 1.0, (grid, grid)).astype(np.float32)

    zoom_y = size / grid
    zoom_x = size / grid
    noise = ndimage.zoom(coarse, (zoom_y, zoom_x), order=3)
    noise = noise[:size, :size].astype(np.float32)

    if blur_sigma > 0:
        noise = ndimage.gaussian_filter(noise, sigma=blur_sigma, mode="reflect")

    noise = noise - float(np.mean(noise))
    std = float(np.std(noise))
    if std > 1e-6:
        noise = noise / std
    return noise.astype(np.float32)


def fbm_noise(
    size: int,
    rng: np.random.Generator,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    base_grid: int = 4,
    blur_sigma: float = 0.0,
) -> Array:
    """
    Fractal Brownian motion built from low-frequency value noise.
    """
    octaves = max(1, int(octaves))
    amp = 1.0
    freq_grid = float(base_grid)
    total = np.zeros((size, size), dtype=np.float32)
    amp_sum = 0.0

    for _ in range(octaves):
        n = value_noise(size, int(round(freq_grid)), rng, blur_sigma=blur_sigma)
        total += n * amp
        amp_sum += amp
        amp *= float(persistence)
        freq_grid *= float(lacunarity)

    total /= max(amp_sum, 1e-6)
    total = total - float(np.mean(total))
    std = float(np.std(total))
    if std > 1e-6:
        total /= std
    return total.astype(np.float32)


def add_macro_noise(h: Array, params: Dict[str, Any], seed_offset: int = 0) -> Array:
    amp = float(params.get("noise_amplitude", 0.0))
    if amp <= 0:
        return h.astype(np.float32, copy=False)

    seed = params.get("seed")
    if seed is not None:
        seed = int(seed) + seed_offset

    rng = _rng(seed)
    n = fbm_noise(
        h.shape[0],
        rng,
        octaves=int(params.get("noise_octaves", 5)),
        persistence=float(params.get("noise_persistence", 0.5)),
        lacunarity=float(params.get("noise_lacunarity", 2.0)),
        base_grid=4,
        blur_sigma=float(params.get("noise_blur_sigma", 8.0)),
    )
    return (h + n * amp).astype(np.float32)


# =============================================================================
# PATH / DISTANCE UTILITIES
# =============================================================================

def bezier_points(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    samples: int = 256,
) -> Array:
    """
    Cubic Bezier points in normalized coordinate space [-1, 1].
    """
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    u = 1.0 - t
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    p3 = np.asarray(p3, dtype=np.float32)

    pts = (
        (u ** 3)[:, None] * p0
        + (3 * u * u * t)[:, None] * p1
        + (3 * u * t * t)[:, None] * p2
        + (t ** 3)[:, None] * p3
    )
    return pts.astype(np.float32)


def rasterize_polyline_mask(
    size: int,
    points: Array,
    thickness_px: int = 2,
) -> Array:
    """
    Rasterizes a normalized-space polyline into a boolean mask.
    """
    mask = np.zeros((size, size), dtype=bool)

    def to_px(p: Array) -> Tuple[int, int]:
        x = int(np.clip((p[0] * 0.5 + 0.5) * (size - 1), 0, size - 1))
        y = int(np.clip((p[1] * 0.5 + 0.5) * (size - 1), 0, size - 1))
        return x, y

    for a, b in zip(points[:-1], points[1:]):
        x0, y0 = to_px(a)
        x1, y1 = to_px(b)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
        xs = np.linspace(x0, x1, steps).astype(np.int32)
        ys = np.linspace(y0, y1, steps).astype(np.int32)
        mask[ys, xs] = True

    if thickness_px > 1:
        structure = ndimage.generate_binary_structure(2, 1)
        mask = ndimage.binary_dilation(mask, structure=structure, iterations=thickness_px)

    return mask


def distance_to_polyline(size: int, points: Array, thickness_px: int = 2) -> Array:
    """
    Fast distance field to a polyline using scipy distance transform.
    Returned distance is normalized to terrain coordinate scale where map width = 2.
    """
    mask = rasterize_polyline_mask(size, points, thickness_px=thickness_px)
    dist_px = ndimage.distance_transform_edt(~mask).astype(np.float32)
    return dist_px * (2.0 / max(size - 1, 1))


# =============================================================================
# PROCEDURAL PRIMITIVES
# =============================================================================

def primitive_radial_bowl(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Playable valley/basin. Good center primitive for "surrounded by mountains".
    """
    base = float(params["base_elevation"])
    center_depth = float(params["center_depth"])
    rim_height = float(params["rim_height"])
    inner = float(params["inner_radius"])
    outer = float(params["outer_radius"])

    bowl = smoothstep(inner, outer, r)
    center_low = 1.0 - smoothstep(0.0, inner, r)

    h = base - center_depth * center_low + rim_height * bowl
    return h.astype(np.float32)


def primitive_mountain_basin(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Basin in center, mountains/rim outside, asymmetric passes.
    """
    cx = float(params.get("basin_center_x", 0.0))
    cy = float(params.get("basin_center_y", 0.0))
    ax = max(float(params.get("basin_aspect_x", 1.0)), 1e-3)
    ay = max(float(params.get("basin_aspect_y", 1.0)), 1e-3)
    dx = (x - cx) / ax
    dy = (y - cy) / ay
    rr = np.sqrt(dx * dx + dy * dy)
    h = primitive_radial_bowl(dx, dy, rr, params)
    base = float(params["base_elevation"])
    rim_height = float(params["rim_height"])

    # Ridge modulation around the ring.
    theta = np.arctan2(dy, dx)
    ridges = (
        0.55 * np.sin(5.0 * theta + float(params.get("ridge_phase_1", 0.8)))
        + 0.30 * np.sin(9.0 * theta - 1.3 + float(params.get("ridge_phase_2", 0.0)))
        + 0.15 * np.sin(14.0 * theta + 2.1)
    )
    ring = smoothstep(0.35, 0.70, rr) * (1.0 - smoothstep(0.95, 1.15, rr))
    h += ridges * rim_height * 0.22 * ring

    # Carve a few lower passes so the ring does not look like a perfect bowl.
    pass1 = np.exp(-((theta - float(params.get("pass_angle_1", 0.30))) ** 2) / 0.045) * ring
    pass2 = np.exp(-((theta - float(params.get("pass_angle_2", -2.35))) ** 2) / 0.055) * ring
    h -= (pass1 + pass2) * rim_height * 0.25

    # Keep center playable but not flat.
    h = np.maximum(h, base - float(params["center_depth"]) * 0.75)
    return h.astype(np.float32)


def primitive_bezier_canyon(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Canyon carved along a Bezier path. The canyon floor is long and narrow,
    not a giant flat square.
    """
    size = x.shape[0]
    base = float(params["base_elevation"]) + float(params["rim_height"]) * 0.55
    canyon_depth = float(params["canyon_depth"]) * float(params.get("canyon_depth_scale", 1.0))
    width = float(params["canyon_width"]) * float(params.get("canyon_width_scale", 1.0))
    steepness = float(params["wall_steepness"])

    angle = math.radians(float(params.get("canyon_angle_degrees", params.get("angle_degrees", 0.0))))
    ca, sa = math.cos(angle), math.sin(angle)

    p0 = np.array([-1.10, float(params.get("canyon_p0_y", -0.35))], dtype=np.float32)
    p1 = np.array([
        float(params.get("canyon_p1_x", -0.45)),
        float(params.get("canyon_p1_y", -0.75)),
    ], dtype=np.float32)
    p2 = np.array([
        float(params.get("canyon_p2_x", 0.35)),
        float(params.get("canyon_p2_y", 0.65)),
    ], dtype=np.float32)
    p3 = np.array([1.10, float(params.get("canyon_p3_y", 0.25))], dtype=np.float32)

    def rot(p: Array) -> Array:
        return np.array([p[0] * ca - p[1] * sa, p[0] * sa + p[1] * ca], dtype=np.float32)

    pts = bezier_points(rot(p0), rot(p1), rot(p2), rot(p3), samples=512)
    d = distance_to_polyline(size, pts, thickness_px=2)

    floor = 1.0 - smoothstep(width * 0.35, width, d)
    wall = smoothstep(width * 0.6, width * steepness, d)

    h = np.full_like(x, base, dtype=np.float32)
    h -= canyon_depth * floor
    h -= canyon_depth * 0.45 * (1.0 - wall)
    h += np.clip(y, -1.0, 1.0) * float(params.get("canyon_rim_bias", 0.0)) * float(params["rim_height"]) * 0.35

    # Add rim shelves/terraces around canyon walls.
    terrace = np.sin(d * 90.0) * np.exp(-d / max(width * 6.0, 1e-4))
    h += terrace * float(params["rim_height"]) * 0.055

    return h.astype(np.float32)


def primitive_coastline(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Ocean-to-land macro shape with a curved coastline.
    y below curve = water side, y above curve = land side, after rotation.
    """
    xr, yr = _rotate(x, y, float(params.get("angle_degrees", 0.0)))
    base = float(params["base_elevation"])
    water_level = float(params["water_level"])
    slope = float(params["slope"])
    amp = float(params["coast_amplitude"])
    freq = float(params["coast_frequency"])
    pos = float(params["coast_position"])

    coast = pos + amp * np.sin(freq * math.pi * xr + 0.55 * np.sin(3.0 * xr))
    signed = yr - coast

    # Negative side is ocean. Positive side rises inland.
    beach_band = smoothstep(-0.08, 0.16, signed)
    inland = smoothstep(0.10, 0.85, signed)

    h = water_level - 18.0 + beach_band * 20.0 + inland * slope
    h += base - water_level

    # Slight dune/backshore ridge inland from beach.
    dune = np.exp(-((signed - 0.20) ** 2) / 0.012)
    h += dune * float(params.get("dune_height", 18.0)) * 0.45

    return h.astype(np.float32)


def primitive_layered_dunes(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Directional dune fields with large rolling sine layers.
    """
    xr, yr = _rotate(x, y, float(params.get("angle_degrees", 25.0)))
    base = float(params["base_elevation"])
    dune_height = float(params["dune_height"])
    freq = float(params["dune_frequency"])

    waves = (
        np.sin((xr * freq + 0.8 * np.sin(yr * 2.7)) * math.pi)
        + 0.45 * np.sin((xr * freq * 0.55 + yr * 1.6) * math.pi + 1.2)
        + 0.25 * np.sin((xr * freq * 1.75 - yr * 0.9) * math.pi - 0.7)
    )

    # Sharpen windward/leeward asymmetry a little.
    dunes = np.tanh(waves * 1.2)
    h = base + dunes * dune_height
    h += smoothstep(0.25, 1.1, r) * dune_height * 0.25
    return h.astype(np.float32)


def primitive_rolling_hills(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Soft broad terrain, useful for forests, plains, meadows, and general fallback.
    """
    base = float(params["base_elevation"])
    height = float(params["hill_height"])

    h = base
    h += height * 0.38 * np.sin(2.2 * math.pi * x + 0.5 * np.sin(2.0 * y))
    h += height * 0.27 * np.sin(2.8 * math.pi * y - 0.7 * np.sin(2.0 * x))
    h += height * 0.18 * np.sin(1.4 * math.pi * (x + y))
    h += height * 0.12 * np.cos(3.1 * math.pi * (x - y))
    return h.astype(np.float32)


def primitive_plains(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Low-relief broad plains / meadow / grassland terrain.

    Unlike rolling_hills, this intentionally keeps elevation variation small.
    It is meant for prompts such as:
      - plains
      - flat plains
      - meadow
      - grassland
      - prairie
      - steppe
      - savanna

    Important:
    This is a macro guide, not final detail. The diffusion pass can add
    small terrain texture later, but the large-scale form should stay usable.
    """
    base = float(params["base_elevation"])

    # Main relief is intentionally small. Defaults are low enough that
    # "plains" does not become hills.
    relief = float(params.get("plains_relief", 5.0))
    broad_relief = float(params.get("plains_broad_relief", 3.0))

    trend_angle = float(params.get("plains_trend_angle", 0.0))
    xr, yr = _rotate(x, y, trend_angle)

    # Very broad undulation.
    h = np.full_like(x, base, dtype=np.float32)
    h += broad_relief * np.sin(0.75 * math.pi * xr + 0.35 * np.sin(1.2 * yr))
    h += broad_relief * 0.65 * np.sin(0.65 * math.pi * yr - 0.25 * np.sin(1.1 * xr))

    # Smaller low-amplitude meadow variation.
    h += relief * 0.35 * np.sin(2.0 * math.pi * (xr + 0.35 * yr))
    h += relief * 0.20 * np.cos(2.4 * math.pi * (yr - 0.25 * xr))
    h += xr * float(params.get("plains_drainage_bias", 0.0))

    return h.astype(np.float32)


def primitive_crater(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Crater/bowl with raised rim and rough outer field.
    """
    base = float(params["base_elevation"])
    radius = float(params["radius"])
    depth = float(params["crater_depth"])
    rim_height = float(params["crater_rim_height"])

    interior = 1.0 - smoothstep(radius * 0.15, radius * 0.85, r)
    rim = np.exp(-((r - radius) ** 2) / max(0.001, (radius * 0.14) ** 2))
    ejecta = np.exp(-((r - radius * 1.25) ** 2) / max(0.001, (radius * 0.55) ** 2))

    h = base - depth * interior + rim_height * rim + rim_height * 0.22 * ejecta
    return h.astype(np.float32)


def primitive_island(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Island with raised interior and sloped shore.
    """
    base = float(params["water_level"]) - 18.0
    height = float(params["island_height"])
    radius = float(params["radius"])

    land = 1.0 - smoothstep(radius * 0.65, radius, r)
    core = 1.0 - smoothstep(0.0, radius * 0.55, r)

    h = base + land * height * 0.55 + core * height * 0.45

    # Uneven shoreline, not a perfect circle.
    theta = np.arctan2(y, x)
    shore_wobble = 0.08 * np.sin(5 * theta) + 0.05 * np.sin(9 * theta + 1.4)
    coast = 1.0 - smoothstep(radius + shore_wobble - 0.05, radius + shore_wobble + 0.06, r)
    h = np.where(coast > 0.03, h, base)

    return h.astype(np.float32)


def primitive_lake_basin(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Lake basin with irregular shoreline and surrounding terrain.

    Designed for prompts such as:
      - calm alpine lake
      - mountain lake
      - alpine tarn
      - loch
      - lake in the mountains

    This produces raw terrain heights. Water masking is still handled later,
    but the basin is shaped so only the intended lake area sits below/near
    water level, with dry shore shelves around it.
    """
    base = float(params["base_elevation"])
    water = float(params["water_level"])

    lake_radius = float(params.get("lake_radius", params.get("radius", 0.28)))
    shore_width = float(params.get("lake_shore_width", 0.08))
    basin_depth = float(params.get("lake_depth", 10.0))
    surround_height = float(params.get("lake_surround_height", 75.0))
    shelf_height = float(params.get("lake_shelf_height", 4.0))

    # Slightly off-center lake so the scene does not always become a bullseye.
    cx = float(params.get("lake_center_x", -0.12))
    cy = float(params.get("lake_center_y", 0.08))

    aspect_x = max(float(params.get("lake_aspect_x", 1.0)), 1e-3)
    aspect_y = max(float(params.get("lake_aspect_y", 1.0)), 1e-3)
    dx = (x - cx) / aspect_x
    dy = (y - cy) / aspect_y
    theta = np.arctan2(dy, dx)
    dist = np.sqrt(dx * dx + dy * dy)

    # Irregular shoreline radius. Keep it low-frequency so it looks natural.
    wobble = (
        0.055 * np.sin(3.0 * theta + float(params.get("lake_wobble_1", 0.7)))
        + 0.035 * np.sin(5.0 * theta + float(params.get("lake_wobble_2", -1.4)))
        + 0.020 * np.sin(8.0 * theta + float(params.get("lake_wobble_3", 2.0)))
    )
    local_radius = lake_radius * (1.0 + wobble)

    def local_smoothstep(edge0: Array, edge1: Array, values: Array) -> Array:
        denom = np.maximum(edge1 - edge0, 1e-6)
        t = np.clip((values - edge0) / denom, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    lake_core = 1.0 - local_smoothstep(local_radius * 0.72, local_radius, dist)
    shore = local_smoothstep(local_radius * 0.86, local_radius + shore_width, dist) * (
        1.0 - local_smoothstep(local_radius + shore_width, local_radius + shore_width * 2.4, dist)
    )

    # Start slightly above water so dry shore can exist.
    h = np.full_like(x, water + shelf_height, dtype=np.float32)
    h += (base - water) * 0.15

    # Lake bed below water.
    h -= basin_depth * lake_core

    # Dry shelf around the lake, gently uneven but mostly usable.
    shelf_band = 1.0 - local_smoothstep(local_radius, local_radius + shore_width * 2.2, dist)
    shelf_band *= local_smoothstep(local_radius * 0.82, local_radius + shore_width, dist)
    h += shelf_band * 1.5

    # Surrounding mountains/ridges rise away from lake.
    away = local_smoothstep(local_radius + shore_width, np.full_like(dist, 1.15, dtype=np.float32), dist)
    h += away * surround_height

    # Alpine ridge modulation around outer terrain.
    ring = local_smoothstep(local_radius + 0.15, np.full_like(dist, 0.95, dtype=np.float32), dist)
    ridges = (
        0.45 * np.sin(4.0 * theta + 0.5)
        + 0.30 * np.sin(7.0 * theta - 1.2)
        + 0.15 * np.sin(11.0 * theta + 1.8)
    )
    h += ring * ridges * surround_height * 0.18

    # Shore should stay close to water level, not become cliffs everywhere.
    h = np.where(shore > 0.05, np.minimum(h, water + shelf_height + 3.0), h)

    # Optional outlet notch so lake scenes have a believable drainage direction.
    outlet_angle = float(params.get("lake_outlet_angle", -0.65))
    angle_delta = np.arctan2(np.sin(theta - outlet_angle), np.cos(theta - outlet_angle))
    outlet = np.exp(-(angle_delta * angle_delta) / 0.018) * local_smoothstep(
        local_radius,
        np.full_like(dist, 1.05, dtype=np.float32),
        dist,
    )
    h -= outlet * surround_height * 0.22

    if float(params.get("lake_island_chance", 0.0)) > 0.55:
        island_x = float(params.get("lake_island_x", 0.0))
        island_y = float(params.get("lake_island_y", 0.0))
        island_radius = float(params.get("lake_island_radius", 0.08))
        island_height = float(params.get("lake_island_height", 8.0))
        idx = (x - island_x) / aspect_x
        idy = (y - island_y) / aspect_y
        island_dist = np.sqrt(idx * idx + idy * idy)
        island = 1.0 - smoothstep(island_radius * 0.65, island_radius, island_dist)
        h += island * island_height
        h = np.where(island > 0.05, np.maximum(h, water + 3.0), h)

    return h.astype(np.float32)


def primitive_river_valley(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Broad valley carved around a continuous river path.
    """
    size = x.shape[0]
    base = float(params["base_elevation"]) + 30.0
    valley_depth = float(params["valley_depth"])
    valley_width = float(params["valley_width"])

    angle = math.radians(float(params.get("river_angle_degrees", 0.0)))
    ca, sa = math.cos(angle), math.sin(angle)

    def rot(px: float, py: float) -> np.ndarray:
        return np.array([px * ca - py * sa, px * sa + py * ca], dtype=np.float32)

    p0 = rot(-1.05, float(params.get("river_p0_y", -0.10)))
    p1 = rot(
        float(params.get("river_p1_x", -0.40)),
        float(params.get("river_p1_y", -0.55)),
    )
    p2 = rot(
        float(params.get("river_p2_x", 0.15)),
        float(params.get("river_p2_y", 0.45)),
    )
    p3 = rot(1.05, float(params.get("river_p3_y", 0.05)))
    pts = bezier_points(p0, p1, p2, p3, samples=512)
    d = distance_to_polyline(size, pts, thickness_px=2)
    meander_scale = float(params.get("river_meander_scale", 1.0))
    side_bias = float(params.get("valley_side_bias", 0.0))

    valley = 1.0 - smoothstep(valley_width * 0.25, valley_width * meander_scale, d)
    broad = 1.0 - smoothstep(valley_width * meander_scale, valley_width * 4.5 * meander_scale, d)

    h = np.full_like(x, base, dtype=np.float32)
    h -= valley_depth * broad
    h -= valley_depth * 0.35 * valley

    # River thalweg, narrow center channel.
    channel = 1.0 - smoothstep(0.012, 0.028, d)
    h -= valley_depth * 0.16 * channel

    # Overall downstream slope.
    h += -x * 10.0
    h += y * side_bias * valley_depth * 0.35
    return h.astype(np.float32)


def primitive_mesa_plateau(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Raised flat-ish mesa/plateau with eroded sides.
    """
    base = float(params["base_elevation"])
    rim_height = float(params["rim_height"])
    radius = float(params["radius"])

    # Use superellipse distance for a less circular mesa.
    xr, yr = _rotate(x, y, float(params.get("angle_degrees", 12.0)))
    super_r = (np.abs(xr / 1.05) ** 4 + np.abs(yr / 0.72) ** 4) ** 0.25

    top = 1.0 - smoothstep(radius * 0.80, radius, super_r)
    shoulder = 1.0 - smoothstep(radius, radius * 1.45, super_r)

    h = base + top * rim_height + shoulder * rim_height * 0.35

    # Erosion grooves down the sides.
    theta = np.arctan2(yr, xr)
    grooves = np.maximum(0.0, np.sin(theta * 18.0 + super_r * 13.0))
    side_mask = smoothstep(radius * 0.78, radius * 1.35, super_r) * (1.0 - smoothstep(radius * 1.35, radius * 1.7, super_r))
    h -= grooves * side_mask * rim_height * 0.11

    return h.astype(np.float32)


def primitive_swamp_basin(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Low wet basin with subtle hummocks, channels, and shallow depressions.
    """
    water = float(params["water_level"])
    base = water + 1.5
    depth = float(params["center_depth"])

    basin = 1.0 - smoothstep(0.2, 0.95, r)
    h = base - basin * depth * 0.35

    # Hummocks and shallow channel-like variation.
    h += 2.0 * np.sin(5.5 * x + 2.0 * np.sin(3.0 * y))
    h += 1.4 * np.sin(6.2 * y + 1.7 * np.sin(2.0 * x))

    # Keep close to water level.
    h = water + np.tanh((h - water) / 5.0) * 5.0
    return h.astype(np.float32)


def primitive_archipelago(x: Array, y: Array, r: Array, params: Dict[str, Any]) -> Array:
    """
    Cluster of island blobs over ocean/bathymetry.
    """
    seed = params.get("seed")
    rng = _rng(None if seed is None else int(seed) + 919)
    size = x.shape[0]

    water = float(params["water_level"])
    h = np.full_like(x, water - 25.0, dtype=np.float32)

    count = int(params.get("island_count", 7))
    for _ in range(count):
        cx = rng.uniform(-0.75, 0.75)
        cy = rng.uniform(-0.75, 0.75)
        rad = rng.uniform(0.12, 0.32)
        height = rng.uniform(25.0, float(params["island_height"]))
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = 1.0 - smoothstep(rad * 0.6, rad, d)
        h += blob * height

    h = ndimage.gaussian_filter(h, sigma=max(2.0, size / 512.0), mode="reflect")
    return h.astype(np.float32)


PRIMITIVES = {
    "radial_bowl": primitive_radial_bowl,
    "mountain_basin": primitive_mountain_basin,
    "bezier_canyon": primitive_bezier_canyon,
    "canyon": primitive_bezier_canyon,
    "coastline": primitive_coastline,
    "beach": primitive_coastline,
    "layered_dunes": primitive_layered_dunes,
    "dunes": primitive_layered_dunes,
    "rolling_hills": primitive_rolling_hills,
    "plains": primitive_plains,
    "plain": primitive_plains,
    "grassland": primitive_plains,
    "meadow": primitive_plains,
    "prairie": primitive_plains,
    "steppe": primitive_plains,
    "savanna": primitive_plains,
    "savannah": primitive_plains,
    "hills": primitive_rolling_hills,
    "crater": primitive_crater,
    "island": primitive_island,
    "lake_basin": primitive_lake_basin,
    "lake": primitive_lake_basin,
    "alpine_lake": primitive_lake_basin,
    "mountain_lake": primitive_lake_basin,
    "tarn": primitive_lake_basin,
    "loch": primitive_lake_basin,
    "river_valley": primitive_river_valley,
    "mesa_plateau": primitive_mesa_plateau,
    "mesa": primitive_mesa_plateau,
    "plateau": primitive_mesa_plateau,
    "swamp_basin": primitive_swamp_basin,
    "swamp": primitive_swamp_basin,
    "archipelago": primitive_archipelago,
}


# =============================================================================
# PUBLIC MACRO GENERATOR
# =============================================================================

def generate_macro_shape(params: ParamsLike) -> Array:
    """
    Generate a low-frequency global macro heightmap.

    Parameters
    ----------
    params:
        JSON string or dict. Important keys:

        base_shape:
            One of:
                radial_bowl
                mountain_basin
                canyon / bezier_canyon
                coastline / beach
                dunes / layered_dunes
                rolling_hills
                crater
                island
                river_valley
                mesa / plateau / mesa_plateau
                swamp / swamp_basin
                archipelago

        size:
            Output width and height. Default 1024.

        base_elevation, center_depth, rim_height, noise_amplitude, seed, etc.

    Returns
    -------
    np.ndarray float32, shape [size, size].
    """
    p = _coerce_params(params)
    size = int(p["size"])
    if size < 32:
        raise ValueError("size must be >= 32")

    x, y, r = _coords(size)

    shape = str(p.get("base_shape", "rolling_hills")).lower().strip()
    primitive = PRIMITIVES.get(shape)
    if primitive is None:
        valid = ", ".join(sorted(PRIMITIVES.keys()))
        raise ValueError(f"Unknown base_shape {shape!r}. Valid shapes: {valid}")

    h = primitive(x, y, r, p)
    h = add_macro_noise(h, p)

    # Final macro blur keeps this as a guide shape, not noisy final terrain.
    final_blur = float(p.get("final_blur_sigma", 1.25))
    if final_blur > 0:
        h = ndimage.gaussian_filter(h, sigma=final_blur, mode="reflect").astype(np.float32)

    return _clamp_height(h, p)


def generate_macro_shape_with_masks(params: ParamsLike) -> Dict[str, Array]:
    """
    Convenience wrapper that returns the heightmap plus simple derived masks.

    This is useful for the rest of the pipeline:
        - water mask
        - slope mask
        - playable-ish flatter mask
    """
    p = _coerce_params(params)
    h = generate_macro_shape(p)

    gy, gx = np.gradient(h)
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    water_level = float(p["water_level"])
    water = h <= water_level
    playable = slope < float(p.get("playable_slope_threshold", 6.0))

    return {
        "height": h.astype(np.float32),
        "slope": slope.astype(np.float32),
        "water": water.astype(np.uint8),
        "playable": playable.astype(np.uint8),
    }


# =============================================================================
# HYDRAULIC EROSION
# =============================================================================

DEFAULT_EROSION_PARAMS: Dict[str, Any] = {
    "iterations": 50000,
    "max_steps": 60,
    "inertia": 0.05,
    "capacity": 4.0,
    "min_capacity": 0.01,
    "deposit_rate": 0.30,
    "erode_rate": 0.30,
    "evaporate_rate": 0.02,
    "gravity": 4.0,
    "initial_water": 1.0,
    "initial_speed": 1.0,
    "erosion_radius": 3,
    "seed": None,
}


@njit(cache=True)
def _bilinear_height_and_gradient(h: Array, x: float, y: float) -> Tuple[float, float, float]:
    """
    Return bilinear-sampled height, gx, gy.

    x and y are in pixel coordinates.
    """
    height_size_y, height_size_x = h.shape

    xi = int(x)
    yi = int(y)

    if xi < 0:
        xi = 0
    if yi < 0:
        yi = 0
    if xi >= height_size_x - 1:
        xi = height_size_x - 2
    if yi >= height_size_y - 1:
        yi = height_size_y - 2

    xf = x - xi
    yf = y - yi

    h00 = h[yi, xi]
    h10 = h[yi, xi + 1]
    h01 = h[yi + 1, xi]
    h11 = h[yi + 1, xi + 1]

    h0 = h00 * (1.0 - xf) + h10 * xf
    h1 = h01 * (1.0 - xf) + h11 * xf
    height = h0 * (1.0 - yf) + h1 * yf

    gx = (h10 - h00) * (1.0 - yf) + (h11 - h01) * yf
    gy = (h01 - h00) * (1.0 - xf) + (h11 - h10) * xf

    return height, gx, gy


@njit(cache=True)
def _deposit_bilinear(h: Array, x: float, y: float, amount: float) -> None:
    height_size_y, height_size_x = h.shape
    xi = int(x)
    yi = int(y)

    if xi < 0 or yi < 0 or xi >= height_size_x - 1 or yi >= height_size_y - 1:
        return

    xf = x - xi
    yf = y - yi

    w00 = (1.0 - xf) * (1.0 - yf)
    w10 = xf * (1.0 - yf)
    w01 = (1.0 - xf) * yf
    w11 = xf * yf

    h[yi, xi] += amount * w00
    h[yi, xi + 1] += amount * w10
    h[yi + 1, xi] += amount * w01
    h[yi + 1, xi + 1] += amount * w11


@njit(cache=True)
def _erode_radius(h: Array, x: float, y: float, amount: float, radius: int) -> None:
    height_size_y, height_size_x = h.shape
    cx = int(x)
    cy = int(y)

    total_weight = 0.0
    r2 = radius * radius

    for oy in range(-radius, radius + 1):
        py = cy + oy
        if py < 0 or py >= height_size_y:
            continue
        for ox in range(-radius, radius + 1):
            px = cx + ox
            if px < 0 or px >= height_size_x:
                continue
            d2 = ox * ox + oy * oy
            if d2 <= r2:
                w = 1.0 - math.sqrt(d2) / max(radius, 1)
                total_weight += w

    if total_weight <= 0.0:
        return

    for oy in range(-radius, radius + 1):
        py = cy + oy
        if py < 0 or py >= height_size_y:
            continue
        for ox in range(-radius, radius + 1):
            px = cx + ox
            if px < 0 or px >= height_size_x:
                continue
            d2 = ox * ox + oy * oy
            if d2 <= r2:
                w = 1.0 - math.sqrt(d2) / max(radius, 1)
                delta = amount * (w / total_weight)
                # Do not erode below zero locally in one hit.
                if delta > h[py, px]:
                    delta = h[py, px]
                h[py, px] -= delta


@njit(cache=True)
def _hydraulic_erosion_numba(
    height: Array,
    iterations: int,
    max_steps: int,
    inertia: float,
    capacity: float,
    min_capacity: float,
    deposit_rate: float,
    erode_rate: float,
    evaporate_rate: float,
    gravity: float,
    initial_water: float,
    initial_speed: float,
    erosion_radius: int,
    seed: int,
) -> Array:
    np.random.seed(seed)

    h = height.copy()
    rows, cols = h.shape

    for _ in range(iterations):
        x = np.random.random() * (cols - 2) + 0.5
        y = np.random.random() * (rows - 2) + 0.5

        dir_x = 0.0
        dir_y = 0.0
        speed = initial_speed
        water = initial_water
        sediment = 0.0

        for _step in range(max_steps):
            old_x = x
            old_y = y

            current_h, grad_x, grad_y = _bilinear_height_and_gradient(h, x, y)

            dir_x = dir_x * inertia - grad_x * (1.0 - inertia)
            dir_y = dir_y * inertia - grad_y * (1.0 - inertia)

            length = math.sqrt(dir_x * dir_x + dir_y * dir_y)
            if length < 1e-8:
                break

            dir_x /= length
            dir_y /= length

            x += dir_x
            y += dir_y

            if x < 1.0 or y < 1.0 or x >= cols - 2 or y >= rows - 2:
                break

            new_h, _, _ = _bilinear_height_and_gradient(h, x, y)
            delta_h = new_h - current_h

            sediment_capacity = max(-delta_h * speed * water * capacity, min_capacity)

            if sediment > sediment_capacity or delta_h > 0.0:
                # Deposit sediment if uphill or over capacity.
                if delta_h > 0.0:
                    amount = min(delta_h, sediment)
                else:
                    amount = (sediment - sediment_capacity) * deposit_rate

                if amount > 0.0:
                    sediment -= amount
                    _deposit_bilinear(h, old_x, old_y, amount)
            else:
                # Erode if under capacity and moving downhill.
                amount = min((sediment_capacity - sediment) * erode_rate, -delta_h)
                if amount > 0.0:
                    _erode_radius(h, old_x, old_y, amount, erosion_radius)
                    sediment += amount

            speed = math.sqrt(max(0.0, speed * speed + delta_h * -gravity))
            water *= (1.0 - evaporate_rate)

            if water <= 1e-5:
                break

    return h


def hydraulic_erosion(
    height: Array,
    erosion_params: Optional[Dict[str, Any]] = None,
) -> Array:
    """
    Particle-based hydraulic erosion.

    This is intended as a fast macro/detail terrain process after the initial
    procedural shape, before AI detail or final material placement.

    Parameters
    ----------
    height:
        2D float array.

    erosion_params:
        Optional dict. Important keys:
            iterations:      more particles = stronger and slower
            max_steps:       path length per particle
            inertia:         high means particles keep direction
            capacity:        sediment carrying capacity multiplier
            deposit_rate:    how fast sediment drops
            erode_rate:      how fast terrain erodes
            evaporate_rate:  water loss per step
            gravity:         acceleration downhill
            erosion_radius:  local erosion brush radius in pixels
            seed:            deterministic seed

    Returns
    -------
    np.ndarray float32, same shape.
    """
    p = dict(DEFAULT_EROSION_PARAMS)
    if erosion_params:
        p.update(erosion_params)

    h = np.asarray(height, dtype=np.float32)
    seed = p.get("seed")
    if seed is None:
        seed = 1337

    out = _hydraulic_erosion_numba(
        h,
        int(p["iterations"]),
        int(p["max_steps"]),
        float(p["inertia"]),
        float(p["capacity"]),
        float(p["min_capacity"]),
        float(p["deposit_rate"]),
        float(p["erode_rate"]),
        float(p["evaporate_rate"]),
        float(p["gravity"]),
        float(p["initial_water"]),
        float(p["initial_speed"]),
        int(p["erosion_radius"]),
        int(seed),
    )
    return out.astype(np.float32)


# =============================================================================
# DEVELOPMENT / CLI TEST
# =============================================================================

def save_debug_preview(height: Array, path: str) -> None:
    """
    Save a quick preview image. Kept dependency-light except matplotlib.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8), dpi=120)
    plt.imshow(height, cmap="terrain")
    plt.colorbar(label="height")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    example = {
        "base_shape": "mountain_basin",
        "size": 1024,
        "base_elevation": 72,
        "center_depth": 18,
        "rim_height": 130,
        "noise_amplitude": 7,
        "seed": 42,
    }

    h0 = generate_macro_shape(example)
    h1 = hydraulic_erosion(h0, {"iterations": 25000, "seed": 42})

    np.save("macro_height.npy", h1)
    save_debug_preview(h1, "macro_height_preview.png")
    print("Wrote macro_height.npy and macro_height_preview.png")
