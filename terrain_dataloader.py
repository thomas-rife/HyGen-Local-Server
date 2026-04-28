"""
Terrain Heightmap DataLoader for Conditional Latent Diffusion + ControlNet Outpainting.

Handles:
  - Sliding-window 256x256 crops from 512x512 region files
  - Global percentile-based heightmap normalization to [-1, 1]
  - Border context extraction for ControlNet edge-matching
  - Random border masking for robust training
  - PER-CROP caption generation (analyzes the actual 256x256 crop, not the 512x512 parent)
  - Biome ID + ocean mask as auxiliary channels
"""

import os
import glob
import json
import random
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL NORMALIZATION CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

HEIGHT_GLOBAL_MIN = -3.0
HEIGHT_GLOBAL_MAX = 280.0


def compute_global_stats(data_dir: str, percentile_lo: float = 0.5, percentile_hi: float = 99.5):
    """One-time utility: scan all height .npy files and print normalization constants."""
    all_vals = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "seed_*", "*.height.npy"))):
        arr = np.load(fp).astype(np.float32).ravel()
        all_vals.append(arr[::16])
    all_vals = np.concatenate(all_vals)
    lo = np.percentile(all_vals, percentile_lo)
    hi = np.percentile(all_vals, percentile_hi)
    print(f"Dataset height stats:")
    print(f"  p{percentile_lo}  = {lo:.1f}")
    print(f"  p{percentile_hi} = {hi:.1f}")
    print(f"  mean    = {all_vals.mean():.1f}")
    print(f"  std     = {all_vals.std():.1f}")
    print(f"\nSet in terrain_dataloader.py:")
    print(f"  HEIGHT_GLOBAL_MIN = {lo:.1f}")
    print(f"  HEIGHT_GLOBAL_MAX = {hi:.1f}")
    return lo, hi


def normalize_height(h: np.ndarray) -> np.ndarray:
    h = (h - HEIGHT_GLOBAL_MIN) / (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN)
    h = h * 2.0 - 1.0
    return np.clip(h, -1.0, 1.0)


def denormalize_height(h: np.ndarray) -> np.ndarray:
    h = (h + 1.0) / 2.0
    h = h * (HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN) + HEIGHT_GLOBAL_MIN
    return h


# ──────────────────────────────────────────────────────────────────────────────
# 2. PER-CROP CAPTION GENERATION
#    Generates a caption from the actual 256x256 crop the model sees,
#    NOT from the parent 512x512 region.
# ──────────────────────────────────────────────────────────────────────────────

OCEAN_BIOME_NAMES = {
    "minecraft:ocean", "minecraft:deep_ocean",
    "minecraft:cold_ocean", "minecraft:deep_cold_ocean",
    "minecraft:frozen_ocean", "minecraft:deep_frozen_ocean",
    "minecraft:lukewarm_ocean", "minecraft:deep_lukewarm_ocean",
    "minecraft:warm_ocean",
}

def _format_biome(biome_id: str) -> str:
    return biome_id.split(':')[-1].replace('_', ' ')


def generate_crop_caption(
    height_crop: np.ndarray,
    biome_crop: np.ndarray,
    id_to_name: dict,
    seed_str: str,
) -> str:
    """
    Generate a caption for a specific 256x256 crop based on ITS terrain
    and ITS biomes — not the parent region's.

    Args:
        height_crop: [256, 256] raw Y-level heightmap (before normalization)
        biome_crop:  [256, 256] uint32 biome ID array
        id_to_name:  dict mapping uint32 biome ID → "minecraft:plains" etc.
        seed_str:    deterministic seed for phrasing variation
    """
    rng = random.Random(seed_str)
    h = height_crop.astype(np.float32)

    # ── Biome counts for THIS crop ──
    unique_ids, counts = np.unique(biome_crop, return_counts=True)
    total = counts.sum()
    crop_biome_counts = {}
    for uid, cnt in zip(unique_ids, counts):
        name = id_to_name.get(int(uid), f"unknown:{uid}")
        crop_biome_counts[name] = int(cnt)

    # Top biomes (>5% of crop)
    sorted_biomes = sorted(crop_biome_counts.items(), key=lambda x: x[1], reverse=True)
    top_biomes = [(_format_biome(b), c / total) for b, c in sorted_biomes if c / total > 0.05][:3]

    # ── Elevation stats ──
    y_min, y_max = int(h.min()), int(h.max())
    y_mean = float(h.mean())
    y_range = y_max - y_min

    # ── Slope/roughness — use elevation std and range as primary signals ──
    # These are more reliable than per-pixel gradients which are noisy
    y_std = float(h.std())

    # ── Elevation description ──
    if y_mean < 50:
        elev = "low-lying"
    elif y_mean < 70:
        elev = "near sea level"
    elif y_mean < 100:
        elev = "moderately elevated"
    elif y_mean < 150:
        elev = "highland"
    else:
        elev = "high-altitude"

    # ── Shape description (based on std + range, not per-pixel slope) ──
    if y_std < 3 and y_range < 15:
        shape = "flat"
    elif y_std < 8 and y_range < 35:
        shape = "gently rolling"
    elif y_std < 15 and y_range < 70:
        shape = "hilly"
    elif y_std < 30 and y_range < 140:
        shape = "rugged"
    else:
        shape = "steep and mountainous"

    # ── Range description ──
    if y_range < 15:
        range_desc = "minimal elevation change"
    elif y_range < 40:
        range_desc = "gentle elevation variation"
    elif y_range < 80:
        range_desc = "moderate elevation variation"
    elif y_range < 150:
        range_desc = "significant elevation variation"
    else:
        range_desc = "dramatic elevation extremes"

    # ── Biome string ──
    if top_biomes:
        names = [n for n, _ in top_biomes]
        if len(names) == 1:
            biome_str = names[0]
        elif len(names) == 2:
            conn = rng.choice(["transitioning into", "bordering", "and"])
            biome_str = f"{names[0]} {conn} {names[1]}"
        else:
            biome_str = f"{names[0]}, {names[1]}, and {names[2]}"
    else:
        biome_str = "mixed terrain"

    # ── Coastline detection ──
    ocean_frac = sum(crop_biome_counts.get(b, 0) for b in OCEAN_BIOME_NAMES) / total
    has_coast = 0.05 < ocean_frac < 0.65

    # ── Spatial gradient ──
    mid = h.shape[0] // 2
    quadrants = {
        "northwest": h[:mid, :mid].mean(),
        "northeast": h[:mid, mid:].mean(),
        "southwest": h[mid:, :mid].mean(),
        "southeast": h[mid:, mid:].mean(),
    }
    sorted_q = sorted(quadrants.items(), key=lambda x: x[1])
    elev_gradient = sorted_q[-1][1] - sorted_q[0][1]

    # ── Assemble caption ──
    templates = [
        f"{shape.capitalize()} {biome_str}, {elev} with {range_desc} (Y {y_min}-{y_max}).",
        f"{elev.capitalize()} {biome_str} terrain. {shape.capitalize()} topography with {range_desc}.",
        f"A region of {biome_str}, {shape} and {elev}, elevation Y {y_min} to {y_max}.",
        f"{biome_str.capitalize()}, {shape} {elev} terrain with {range_desc}.",
    ]
    caption = rng.choice(templates)

    if elev_gradient > 25:
        highest_dir = sorted_q[-1][0]
        lowest_dir = sorted_q[0][0]
        spatial_phrases = [
            f" Rising toward the {highest_dir}.",
            f" Higher in the {highest_dir}, lower in the {lowest_dir}.",
            f" Elevation increases toward the {highest_dir}.",
        ]
        caption += rng.choice(spatial_phrases)

    if has_coast:
        caption += " A coastline divides land and ocean."

    return caption


# ──────────────────────────────────────────────────────────────────────────────
# 3. REGION INDEX
# ──────────────────────────────────────────────────────────────────────────────

class RegionIndex:
    """Scans a directory for region file groups."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.regions: List[Dict[str, str]] = []
        self._scan()

    def _scan(self):
        height_files = sorted(glob.glob(os.path.join(self.data_dir, "seed_*", "r.*.height.npy")))
        for hf in height_files:
            region_folder = os.path.dirname(hf)
            basename = os.path.basename(hf)
            prefix = basename.replace(".height.npy", "")

            biome_f   = os.path.join(region_folder, f"{prefix}.biome.npy")
            ocean_f   = os.path.join(region_folder, f"{prefix}.ocean_mask.npy")
            meta_f    = os.path.join(region_folder, f"{prefix}.meta.json")

            # Caption file is now optional — we generate captions per-crop
            if not all(os.path.exists(f) for f in [biome_f, ocean_f, meta_f]):
                continue

            # Load biome ID→name mapping from metadata
            with open(meta_f, 'r') as f:
                meta = json.load(f)
            # biome_ids in meta: {"minecraft:plains": 12345, ...}
            # We need the reverse: {12345: "minecraft:plains", ...}
            id_to_name = {v: k for k, v in meta.get("biome_ids", {}).items()}

            self.regions.append({
                "height":     hf,
                "biome":      biome_f,
                "ocean_mask": ocean_f,
                "id_to_name": id_to_name,
                "prefix":     prefix,
                "folder":     region_folder,
            })

    def __len__(self):
        return len(self.regions)


# ──────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ──────────────────────────────────────────────────────────────────────────────

class TerrainDataset(Dataset):
    """
    Yields 256x256 crops with per-crop captions.

    Each sample is a dict:
        height_crop   : [1, 256, 256] float32, normalized to [-1, 1]
        biome_crop    : [1, 256, 256] int64 biome IDs
        ocean_crop    : [1, 256, 256] float32 binary mask
        border_context: [1, 256, 256] float32
        border_mask   : [1, 256, 256] float32
        caption       : str (generated from THIS crop's terrain, not the parent region)
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 256,
        stride: int = 128,
        border_width: int = 16,
        min_active_borders: int = 0,
        max_active_borders: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.stride = stride
        self.border_width = border_width
        self.min_active_borders = min_active_borders
        self.max_active_borders = max_active_borders
        self.augment = augment

        self.index = RegionIndex(data_dir)
        assert len(self.index) > 0, f"No valid regions found in {data_dir}"

        self.crops: List[Tuple[int, int, int]] = []
        region_size = 512
        for ri in range(len(self.index)):
            for r in range(0, region_size - crop_size + 1, stride):
                for c in range(0, region_size - crop_size + 1, stride):
                    self.crops.append((ri, r, c))

    def __len__(self):
        return len(self.crops)

    def _load_region(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Load region data. Returns (height, biome, ocean, id_to_name)."""
        reg = self.index.regions[idx]
        height = np.load(reg["height"]).astype(np.float32)
        biome  = np.load(reg["biome"]).astype(np.int64)
        ocean  = np.load(reg["ocean_mask"]).astype(np.float32)
        return height, biome, ocean, reg["id_to_name"]

    def _extract_border_context(
        self, height_full: np.ndarray, r: int, c: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        cs = self.crop_size
        bw = self.border_width
        H, W = height_full.shape

        context = np.zeros((cs, cs), dtype=np.float32)
        mask    = np.zeros((cs, cs), dtype=np.float32)

        available = []
        if r >= bw:              available.append("top")
        if r + cs + bw <= H:     available.append("bottom")
        if c >= bw:              available.append("left")
        if c + cs + bw <= W:     available.append("right")

        n_active = random.randint(self.min_active_borders,
                                  min(self.max_active_borders, len(available)))
        active = random.sample(available, k=n_active) if n_active > 0 else []

        if "top" in active:
            strip = height_full[r - bw : r, c : c + cs]
            context[:bw, :] = normalize_height(strip)
            mask[:bw, :] = 1.0
        if "bottom" in active:
            strip = height_full[r + cs : r + cs + bw, c : c + cs]
            context[cs - bw :, :] = normalize_height(strip)
            mask[cs - bw :, :] = 1.0
        if "left" in active:
            strip = height_full[r : r + cs, c - bw : c]
            context[:, :bw] = normalize_height(strip)
            mask[:, :bw] = 1.0
        if "right" in active:
            strip = height_full[r : r + cs, c + cs : c + cs + bw]
            context[:, cs - bw :] = normalize_height(strip)
            mask[:, cs - bw :] = 1.0

        return context, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        region_idx, r, c = self.crops[idx]
        height_full, biome_full, ocean_full, id_to_name = self._load_region(region_idx)

        cs = self.crop_size

        # --- Crop extraction ---
        height_crop = height_full[r : r + cs, c : c + cs].copy()
        biome_crop  = biome_full[r : r + cs, c : c + cs].copy()
        ocean_crop  = ocean_full[r : r + cs, c : c + cs].copy()

        # --- Generate caption for THIS specific crop ---
        # Use region+position as deterministic seed so same crop = same caption
        reg = self.index.regions[region_idx]
        caption_seed = f"{reg['folder']}/{reg['prefix']}_{r}_{c}"
        caption = generate_crop_caption(
            height_crop, biome_crop.astype(np.uint32), id_to_name, caption_seed
        )

        # --- Border context ---
        border_context, border_mask = self._extract_border_context(height_full, r, c)

        # --- Augmentation ---
        if self.augment:
            k = random.randint(0, 3)
            if k > 0:
                height_crop    = np.rot90(height_crop, k).copy()
                biome_crop     = np.rot90(biome_crop, k).copy()
                ocean_crop     = np.rot90(ocean_crop, k).copy()
                border_context = np.rot90(border_context, k).copy()
                border_mask    = np.rot90(border_mask, k).copy()

            if random.random() > 0.5:
                height_crop    = np.fliplr(height_crop).copy()
                biome_crop     = np.fliplr(biome_crop).copy()
                ocean_crop     = np.fliplr(ocean_crop).copy()
                border_context = np.fliplr(border_context).copy()
                border_mask    = np.fliplr(border_mask).copy()

        # --- Normalize + tensorize ---
        height_norm = normalize_height(height_crop)

        return {
            "height_crop":    torch.from_numpy(height_norm).unsqueeze(0),
            "biome_crop":     torch.from_numpy(biome_crop).unsqueeze(0),
            "ocean_crop":     torch.from_numpy(ocean_crop).unsqueeze(0),
            "border_context": torch.from_numpy(border_context).unsqueeze(0),
            "border_mask":    torch.from_numpy(border_mask).unsqueeze(0),
            "caption":        caption,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5. COLLATE + DATALOADER FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def terrain_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "height_crop":    torch.stack([b["height_crop"] for b in batch]),
        "biome_crop":     torch.stack([b["biome_crop"] for b in batch]),
        "ocean_crop":     torch.stack([b["ocean_crop"] for b in batch]),
        "border_context": torch.stack([b["border_context"] for b in batch]),
        "border_mask":    torch.stack([b["border_mask"] for b in batch]),
        "caption":        [b["caption"] for b in batch],
    }


def create_terrain_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 256,
    stride: int = 128,
    border_width: int = 16,
    augment: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TerrainDataset(
        data_dir=data_dir,
        crop_size=crop_size,
        stride=stride,
        border_width=border_width,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=terrain_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/regions"

    if "--stats" in sys.argv:
        compute_global_stats(data_dir)
        sys.exit(0)

    print(f"Loading dataset from: {data_dir}")
    loader = create_terrain_dataloader(data_dir, batch_size=4, num_workers=0)
    print(f"Dataset size: {len(loader.dataset)} crops")
    print(f"Batches per epoch: {len(loader)}")

    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s} -> {v.shape}  dtype={v.dtype}  range=[{v.min():.2f}, {v.max():.2f}]")
        else:
            print(f"  {k:20s} -> list[{len(v)}]")
            for i, cap in enumerate(v[:4]):
                print(f"    [{i}] {cap}")
