"""
terrain_gui.py — Minimal tkinter GUI for the terrain generation pipeline.

Wraps the img2img pipeline so you can just type a prompt and click Generate.
Models load once on first generation and stay in memory.

What's new in this revision:
  * Prompt is parsed locally (no LLM, no network) into macro parameters,
    then a macro scene adapter builds the global macro heightmap before
    the AI runs.
  * The AI runs as an img2img refinement on top of the macro.
  * Two new sliders: `img2img_strength` (default 0.4) and `overlap`
    (default 64) so you can A/B them.
  * Stitching uses cosine crossfade over overlapping chunks.

Requirements:
  - tkinter (ships with Python on most systems)
  - Pillow  (pip install pillow)   ← for the preview pane

Run:
  python terrain_gui.py
"""

import os
import io
import json
import queue
import secrets
import threading
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

# Heavy deps (torch, transformers, matplotlib) are imported lazily inside the
# worker so the GUI window opens fast.


CONFIG_PATH = Path.home() / ".terrain_gui_config.json"
DEFAULT_CONFIG = {
    "vae_ckpt":          "",
    "unet_ckpt":         "",
    "controlnet_ckpt":   "",
    "last_output_dir":   str(Path.home()),
    "prompt":            "alpine mountains",
    "grid_size":         3,
    "cfg_scale":         5.0,
    "num_steps":         30,        # img2img defaults lower than from-scratch
    "img2img_strength":  0.4,
    "overlap":           64,
}

# Directories (relative to the script's folder) we'll scan for checkpoints.
AUTODETECT_DIRS = [".", "checkpoints", "ckpt", "ckpts", "models", "weights"]

# Patterns used to guess which checkpoint is which. Order = priority.
AUTODETECT_PATTERNS = {
    "controlnet_ckpt": ["controlnet_ema_final", "controlnet_ema", "controlnet"],
    "unet_ckpt":       ["unet_ema_final",       "unet_ema",       "unet"],
    "vae_ckpt":        ["vae_ema_final",        "vae_ema",        "vae"],
}


def resolve_generation_seed(seed):
    if seed is not None:
        return int(seed) & 0xFFFFFFFF
    return int(secrets.randbelow(2 ** 32))


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT AUTO-DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _script_dir() -> Path:
    """Directory containing this script — where we look for checkpoints."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def autodetect_checkpoints() -> dict:
    """Scan the script directory + a few common subfolders for .pt/.pth files
    and try to match them to VAE / U-Net / ControlNet."""
    base = _script_dir()
    candidates = []
    seen = set()
    for sub in AUTODETECT_DIRS:
        d = (base / sub).resolve()
        if not d.is_dir():
            continue
        for pat in ("*.pt", "*.pth"):
            for fp in d.glob(pat):
                if fp in seen:
                    continue
                seen.add(fp)
                candidates.append(fp)

    results = {k: "" for k in AUTODETECT_PATTERNS}
    for role, patterns in AUTODETECT_PATTERNS.items():
        best = None
        best_rank = None
        for fp in candidates:
            name_lower = fp.name.lower()
            for pi, pat in enumerate(patterns):
                if pat in name_lower:
                    rank = (pi, -fp.stat().st_mtime)
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best = fp
                    break
        if best is not None:
            results[role] = str(best)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """Saved config, then drop stale paths, then auto-detect for any blanks."""
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    for key in ("vae_ckpt", "unet_ckpt", "controlnet_ckpt"):
        p = cfg.get(key, "")
        if p and not Path(p).exists():
            cfg[key] = ""

    detected = autodetect_checkpoints()
    for key, found in detected.items():
        if not cfg.get(key) and found:
            cfg[key] = found
    return cfg


def save_config(cfg: dict) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE STDOUT → QUEUE
# ══════════════════════════════════════════════════════════════════════════════

class QueueWriter:
    """File-like object that pushes each write into a queue as a log message."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(("log", line))
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.q.put(("log", self._buf))
            self._buf = ""


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION WORKER
# ══════════════════════════════════════════════════════════════════════════════

class GenerationWorker:
    """Runs model loading + generation on a background thread.

    Communicates back to the GUI via a queue:
        ("log",      str)
        ("status",   str)
        ("preview",  PIL.Image)
        ("diag",     dict)
        ("done",     dict)
        ("error",    str)
        ("busy",     bool)
    """

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self._models = None
        self._loaded_ckpt_paths = None

    def run_generation(self, params: dict) -> None:
        thread = threading.Thread(
            target=self._worker_entry, args=(params,), daemon=True
        )
        thread.start()

    def _worker_entry(self, params: dict) -> None:
        self.q.put(("busy", True))
        try:
            writer = QueueWriter(self.q)
            with redirect_stdout(writer):
                self._do_generation(params)
            writer.flush()
        except Exception as e:
            self.q.put(("error", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))
        finally:
            self.q.put(("busy", False))

    def _do_generation(self, params: dict) -> None:
        # Lazy imports inside the worker — GUI start-up stays fast.
        import torch
        import matplotlib
        matplotlib.use("Agg")
        from PIL import Image

        # Legacy module: only used for load_models, T5TextEncoder,
        # CosineNoiseSchedule, and the prompt-pick / spiral helpers.
        import generate_terrain as gt_legacy
        # New module: the img2img sampler + crossfade stitcher.
        import generate_terrain_img2img as gt
        import terrain_macro
        from terrain_dataloader import denormalize_height
        from macro_prompt_parser import parse_prompt_to_macro_params, summarize
        from terrain_scene_adapter import (
            build_normalized_macro_world,
            apply_constraint_repair,
        )
        from terrain_validation import (
            validate_macro,
            validate_final_package_inputs,
            format_validation_lines,
        )

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps" if torch.mps.is_available() else "cpu"

        self.q.put(("status", f"Device: {device}"))

        ckpt_paths = (
            params["vae_ckpt"], params["unet_ckpt"], params["controlnet_ckpt"]
        )
        seed = resolve_generation_seed(params.get("seed"))
        print(f"[macro] using seed={seed}")
        for p in ckpt_paths:
            if not p or not Path(p).exists():
                raise FileNotFoundError(f"Checkpoint not found: {p!r}")

        # ── Load models (once, cached) ──
        if self._models is None or self._loaded_ckpt_paths != ckpt_paths:
            self.q.put(("status",
                        "Loading models — this can take 30–60s the first time..."))
            vae, unet, controlnet = gt_legacy.load_models(
                params["vae_ckpt"], params["unet_ckpt"], params["controlnet_ckpt"],
                device,
            )
            text_encoder = gt_legacy.T5TextEncoder(device=device)
            schedule = gt_legacy.CosineNoiseSchedule(num_timesteps=1000)
            self._models = (vae, unet, controlnet, text_encoder, schedule)
            self._loaded_ckpt_paths = ckpt_paths
            self.q.put(("status", "Models loaded."))
        else:
            self.q.put(("status", "Using cached models."))

        vae, unet, controlnet, text_encoder, schedule = self._models

        # ── Build prompt spec ──
        directional = {
            "north": params.get("prompt_north") or None,
            "south": params.get("prompt_south") or None,
            "east":  params.get("prompt_east")  or None,
            "west":  params.get("prompt_west")  or None,
        }
        any_directional = any(v for v in directional.values())

        if any_directional and not params.get("use_single_prompt", True):
            single_prompt = None
        else:
            single_prompt = params["prompt"]
            directional = {k: None for k in directional}

        grid_size = int(params["grid_size"])
        if grid_size % 2 == 0:
            raise ValueError(f"grid_size must be odd (got {grid_size}).")

        overlap          = int(params["overlap"])
        img2img_strength = float(params["img2img_strength"])
        if not (0.0 <= img2img_strength <= 1.0):
            raise ValueError(
                f"img2img_strength must be in [0, 1] (got {img2img_strength})"
            )
        if not (0 <= overlap < 256):
            raise ValueError(f"overlap must be in [0, 256) (got {overlap})")

        # ── Parse prompt → macro params (offline, no LLM) ──
        self.q.put(("status", "Parsing prompt → macro parameters..."))
        macro_params = parse_prompt_to_macro_params(
            single_prompt,
            grid_size=grid_size,
            overlap=overlap,
            seed=seed,
            directional=directional if any_directional else None,
        )

        # Adapter: parser schema → terrain_macro schema.
        # The parser returns "primitive"; terrain_macro expects "base_shape".




        # ── Build the global macro heightmap (procedural) ──
        self.q.put(("status",
                    f"Building macro shape (world={macro_params['world_size']}px)..."))

        # terrain_macro returns raw Y-ish heights, not normalized [-1, 1].
        macro_world, macro_params_adapted, macro_stats = build_normalized_macro_world(
            terrain_macro,
            macro_params,
            base_y=63,
            sea_level=63,
        )
        macro_raw_for_validation = macro_params_adapted.get("_macro_raw_with_playable")
        if macro_raw_for_validation is not None:
            macro_report = validate_macro(
                macro_raw_for_validation,
                macro_params_adapted,
                sea_level=63,
            )
            for line in format_validation_lines(macro_report, "[macro-validation]"):
                print(line)

            if macro_report.errors:
                raise ValueError("; ".join(macro_report.errors))
        print(
            f"[macro] {summarize(macro_params)} "
            f"base_shape={macro_params_adapted.get('base_shape')} "
            f"playable=({macro_stats['playable_center_r']:.0f},{macro_stats['playable_center_c']:.0f}) "
            f"target={macro_stats['playable_target']:.2f} "
            f"protect={macro_stats.get('protect_max', 0.0):.2f} "
            f"water_cells={macro_stats.get('water_mask_cells', 0.0):.0f} "
            f"river_cells={macro_stats.get('river_mask_cells', 0.0):.0f} "
            f"feature_water={macro_stats.get('feature_water_cells', 0.0):.0f} "
            f"feature_playable={macro_stats.get('feature_playable_cells', 0.0):.0f} "
            f"raw=({macro_stats['raw_min']:.2f},{macro_stats['raw_max']:.2f}) "
            f"norm=({macro_stats['norm_min']:.3f},{macro_stats['norm_max']:.3f})"
        )

        # ── img2img refinement ──
        t_start = int(round(img2img_strength * (schedule.T - 1)))
        self.q.put(("status",
                    f"Generating {grid_size}×{grid_size} grid "
                    f"(strength={img2img_strength:.2f}, t_start={t_start})..."))

        chunks, _prompts_used, origins_y, origins_x = gt.generate_grid(
            grid_size=grid_size,
            macro_world=macro_world,
            vae=vae, unet=unet, controlnet=controlnet,
            schedule=schedule, text_encoder=text_encoder,
            device=device,
            cfg_scale=float(params["cfg_scale"]),
            num_steps=int(params["num_steps"]),
            single_prompt=single_prompt,
            directional=directional,
            seed=seed,
            overlap=overlap,
            start_timestep=t_start,
            pick_prompt_for_chunk_fn=gt_legacy.pick_prompt_for_chunk,
            spiral_order_fn=gt_legacy.spiral_order,
        )

        # ── Crossfade stitch ──
        stitched_norm = gt.stitch_chunks_crossfade(
            chunks, grid_size,
            origins_y=origins_y, origins_x=origins_x,
            overlap=overlap,
        )
        stitched_raw = denormalize_height(stitched_norm)

        macro_raw_with_playable = macro_params_adapted.get("_macro_raw_with_playable")

        if macro_raw_with_playable is not None:
            stitched_raw = apply_constraint_repair(
                stitched_raw,
                macro_raw_with_playable,
                macro_params_adapted,
                sea_level=63,
                strength=0.78,
            )

            playable_meta = macro_params_adapted.get("_playable_area") or {}
            print(
                f"[macro] constraints repaired "
                f"playable=({playable_meta.get('center_r')},"
                f"{playable_meta.get('center_c')}) "
                f"target={float(playable_meta.get('target_height', 0.0)):.2f}"
            )

        # ── Diagnostics (re-implemented locally; the old function lived in
        #    the legacy generate_terrain.py). With crossfade stitching the
        #    seam L1 should be near-zero — useful as a sanity check. ──
        final_report = validate_final_package_inputs(
            stitched_raw,
            macro_params_adapted,
            sea_level=63,
        )
        for line in format_validation_lines(final_report, "[final-validation]"):
            print(line)

        if final_report.errors:
            raise ValueError("; ".join(final_report.errors))

        diag = self._compute_seam_diagnostics(
            stitched_raw, grid_size, origins_y, origins_x,
        )
        self.q.put(("diag", diag))

        # ── Render PIL preview ──
        img = self._render_pil_image(
            stitched_raw, grid_size, single_prompt, directional,
            origins_y=origins_y, origins_x=origins_x,
        )
        self.q.put(("preview", img))

        # ── Hand the raw heightmap back to the GUI ──
        self.q.put(("done", {
            "image":          img,
            "stitched_raw":   stitched_raw,
            "grid_size":      grid_size,
            "prompt":         single_prompt,
            "directional":    directional,
            "seed":           seed,
            "macro_params":   macro_params_adapted,
            "img2img_strength": img2img_strength,
            "overlap":        overlap,
        }))

    @staticmethod
    def _compute_seam_diagnostics(stitched_raw, grid_size,
                                  origins_y, origins_x):
        """Mean L1 across each chunk-vs-chunk interior seam at the *centre* of
        the overlap region. With cosine crossfade these should be tiny.
        """
        from generate_terrain_img2img import CHUNK_SIZE

        diffs_h = []
        diffs_v = []
        for r in range(grid_size):
            for c in range(grid_size):
                y0, x0 = origins_y[r], origins_x[c]
                # Right-neighbour seam: probe the column at the chunk's right
                # edge (x0 + CHUNK_SIZE - 1) vs the next column.
                if c + 1 < grid_size:
                    col = x0 + CHUNK_SIZE - 1
                    if col + 1 < stitched_raw.shape[1]:
                        a = stitched_raw[y0:y0 + CHUNK_SIZE, col]
                        b = stitched_raw[y0:y0 + CHUNK_SIZE, col + 1]
                        diffs_v.append(float(np.mean(np.abs(a - b))))
                if r + 1 < grid_size:
                    row = y0 + CHUNK_SIZE - 1
                    if row + 1 < stitched_raw.shape[0]:
                        a = stitched_raw[row,     x0:x0 + CHUNK_SIZE]
                        b = stitched_raw[row + 1, x0:x0 + CHUNK_SIZE]
                        diffs_h.append(float(np.mean(np.abs(a - b))))

        all_d = diffs_h + diffs_v
        return {
            "num_seams":     len(all_d),
            "mean_l1":       float(np.mean(all_d))     if all_d  else 0.0,
            "max_l1":        float(np.max(all_d))      if all_d  else 0.0,
            "median_l1":     float(np.median(all_d))   if all_d  else 0.0,
            "horiz_mean_l1": float(np.mean(diffs_h))   if diffs_h else 0.0,
            "vert_mean_l1":  float(np.mean(diffs_v))   if diffs_v else 0.0,
        }

    @staticmethod
    def _render_pil_image(stitched_raw, grid_size, single_prompt, directional,
                          origins_y, origins_x):
        """Render with matplotlib → PIL Image (in-memory)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        from terrain_dataloader import HEIGHT_GLOBAL_MIN, HEIGHT_GLOBAL_MAX
        from generate_terrain_img2img import CHUNK_SIZE

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        im = ax.imshow(
            stitched_raw, cmap="terrain",
            vmin=HEIGHT_GLOBAL_MIN, vmax=HEIGHT_GLOBAL_MAX,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Y-level")

        # Overlay chunk boundaries at their actual (overlap-aware) origins
        H = stitched_raw.shape[0]
        for x0 in origins_x[1:]:
            ax.axvline(x0 - 0.5, color="white", alpha=0.25, linewidth=0.6)
        for y0 in origins_y[1:]:
            ax.axhline(y0 - 0.5, color="white", alpha=0.25, linewidth=0.6)

        # Mark the centre chunk (where the player spawns).
        cr = cc = grid_size // 2
        ax.add_patch(Rectangle(
            (origins_x[cc] - 0.5, origins_y[cr] - 0.5),
            CHUNK_SIZE, CHUNK_SIZE,
            linewidth=1.5, edgecolor="red", facecolor="none",
        ))

        if single_prompt:
            title = f"{grid_size}×{grid_size}  \"{single_prompt}\""
        else:
            parts = [f"{k[0].upper()}:{v[:12]}"
                     for k, v in directional.items() if v]
            title = f"{grid_size}×{grid_size}  " + " / ".join(parts)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()


# ══════════════════════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════════════════════

class TerrainGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        self.worker = GenerationWorker()
        self._last_result = None
        self._preview_tk_image = None

        root.title("Terrain Generator (img2img)")
        root.geometry("1200x820")

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_controls(main)
        self._build_preview(main)

        self.root.after(80, self._poll_queue)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._report_autodetect()

    # ── Layout: controls column ──────────────────────────────────────────────
    def _build_controls(self, parent):
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        # ── Checkpoint files ──
        ck = ttk.LabelFrame(left, text="Checkpoints", padding=8)
        ck.pack(fill="x", pady=(0, 8))
        self.vae_var  = tk.StringVar(value=self.cfg["vae_ckpt"])
        self.unet_var = tk.StringVar(value=self.cfg["unet_ckpt"])
        self.cn_var   = tk.StringVar(value=self.cfg["controlnet_ckpt"])
        self._file_row(ck, "VAE",        self.vae_var,  0)
        self._file_row(ck, "U-Net",      self.unet_var, 1)
        self._file_row(ck, "ControlNet", self.cn_var,   2)

        # ── Prompt ──
        pf = ttk.LabelFrame(left, text="Prompt", padding=8)
        pf.pack(fill="x", pady=(0, 8))
        ttk.Label(pf, text="Main prompt:").grid(row=0, column=0, sticky="w")
        self.prompt_var = tk.StringVar(value=self.cfg["prompt"])
        self.prompt_entry = ttk.Entry(pf, textvariable=self.prompt_var, width=46)
        self.prompt_entry.grid(row=0, column=1, sticky="we", padx=(4, 0), pady=2)
        pf.columnconfigure(1, weight=1)

        self.use_directional = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pf, text="Use directional prompts (overrides main prompt)",
            variable=self.use_directional,
            command=self._toggle_directional,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 2))

        self.north_var = tk.StringVar()
        self.south_var = tk.StringVar()
        self.east_var  = tk.StringVar()
        self.west_var  = tk.StringVar()
        self._dir_rows = []
        for i, (label, var) in enumerate([
            ("North", self.north_var), ("South", self.south_var),
            ("East",  self.east_var),  ("West",  self.west_var),
        ]):
            lbl = ttk.Label(pf, text=f"{label}:")
            lbl.grid(row=2 + i, column=0, sticky="w")
            ent = ttk.Entry(pf, textvariable=var, width=46)
            ent.grid(row=2 + i, column=1, sticky="we", padx=(4, 0), pady=1)
            self._dir_rows.append((lbl, ent))
        self._toggle_directional()

        # ── Settings ──
        sf = ttk.LabelFrame(left, text="Settings", padding=8)
        sf.pack(fill="x", pady=(0, 8))
        sf.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(sf, text="Grid size (odd):").grid(row=row, column=0, sticky="w")
        self.grid_var = tk.IntVar(value=self.cfg["grid_size"])
        ttk.Spinbox(
            sf, from_=1, to=11, increment=2, textvariable=self.grid_var, width=6
        ).grid(row=row, column=1, sticky="w", padx=(4, 0))
        row += 1

        ttk.Label(sf, text="CFG scale:").grid(row=row, column=0, sticky="w")
        self.cfg_scale_var = tk.DoubleVar(value=self.cfg["cfg_scale"])
        ttk.Spinbox(
            sf, from_=1.0, to=15.0, increment=0.5,
            textvariable=self.cfg_scale_var, width=6, format="%.1f",
        ).grid(row=row, column=1, sticky="w", padx=(4, 0))
        row += 1

        ttk.Label(sf, text="DDIM steps:").grid(row=row, column=0, sticky="w")
        self.steps_var = tk.IntVar(value=self.cfg["num_steps"])
        ttk.Spinbox(
            sf, from_=10, to=200, increment=5,
            textvariable=self.steps_var, width=6,
        ).grid(row=row, column=1, sticky="w", padx=(4, 0))
        row += 1

        # ── img2img controls ──────────────────────────────────────────────
        ttk.Separator(sf, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(8, 4)
        )
        row += 1

        # img2img strength: slider + numeric readout
        ttk.Label(sf, text="img2img strength:").grid(row=row, column=0, sticky="w")
        self.strength_var = tk.DoubleVar(value=self.cfg["img2img_strength"])
        self.strength_label_var = tk.StringVar(
            value=f"{self.strength_var.get():.2f}"
        )
        strength_scale = ttk.Scale(
            sf, from_=0.0, to=1.0, orient="horizontal",
            variable=self.strength_var,
            command=lambda v: self.strength_label_var.set(f"{float(v):.2f}"),
        )
        strength_scale.grid(row=row, column=1, sticky="we", padx=(4, 4))
        ttk.Label(sf, textvariable=self.strength_label_var, width=5,
                  anchor="e").grid(row=row, column=2, sticky="w")
        row += 1

        ttk.Label(sf, text="(0 = pure macro · 1 = ignore macro)",
                  foreground="#666", font=("TkDefaultFont", 8)).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=(0, 0)
        )
        row += 1

        # Overlap (in pixels): spinbox is plenty
        ttk.Label(sf, text="Chunk overlap (px):").grid(row=row, column=0, sticky="w")
        self.overlap_var = tk.IntVar(value=self.cfg["overlap"])
        ttk.Spinbox(
            sf, from_=0, to=128, increment=8,
            textvariable=self.overlap_var, width=6,
        ).grid(row=row, column=1, sticky="w", padx=(4, 0))
        row += 1

        ttk.Label(sf, text="(higher = smoother seams, smaller world)",
                  foreground="#666", font=("TkDefaultFont", 8)).grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        ttk.Label(sf, text="Seed (blank = random):").grid(row=row, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(sf, textvariable=self.seed_var, width=10).grid(
            row=row, column=1, sticky="w", padx=(4, 0)
        )

        # ── Buttons ──
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=(4, 8))
        self.generate_btn = ttk.Button(
            bf, text="Generate", command=self._on_generate
        )
        self.generate_btn.pack(side="left", fill="x", expand=True)
        self.save_png_btn = ttk.Button(
            bf, text="Save PNG…", command=self._on_save_png, state="disabled"
        )
        self.save_png_btn.pack(side="left", padx=(6, 0))
        self.save_npy_btn = ttk.Button(
            bf, text="Save .npy…", command=self._on_save_npy, state="disabled"
        )
        self.save_npy_btn.pack(side="left", padx=(6, 0))
        self.export_pkg_btn = ttk.Button(
            bf, text="Export Java package…",
            command=self._on_export_package, state="disabled",
        )
        self.export_pkg_btn.pack(side="left", padx=(6, 0))

        # ── Seam diagnostics ──
        df = ttk.LabelFrame(left, text="Seam diagnostics (raw Y-levels)", padding=8)
        df.pack(fill="x", pady=(0, 8))
        self.diag_var = tk.StringVar(value="(no run yet)")
        ttk.Label(df, textvariable=self.diag_var, justify="left",
                  font=("TkFixedFont", 9)).pack(anchor="w")

        # ── Log ──
        lf = ttk.LabelFrame(left, text="Log", padding=4)
        lf.pack(fill="both", expand=True)
        self.log_text = tk.Text(lf, width=54, height=12, font=("TkFixedFont", 9))
        self.log_text.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(lf, command=self.log_text.yview)
        sb.pack(fill="y", side="right")
        self.log_text.config(yscrollcommand=sb.set, state="disabled")

    def _file_row(self, parent, label, var, row):
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=1)
        ent = ttk.Entry(parent, textvariable=var, width=36)
        ent.grid(row=row, column=1, sticky="we", padx=(4, 4), pady=1)
        btn = ttk.Button(
            parent, text="…", width=2,
            command=lambda v=var: self._browse_ckpt(v),
        )
        btn.grid(row=row, column=2, pady=1)
        parent.columnconfigure(1, weight=1)

    def _build_preview(self, parent):
        right = ttk.LabelFrame(parent, text="Preview", padding=4)
        right.grid(row=0, column=1, sticky="nsew")
        self.preview_label = ttk.Label(
            right,
            text="Output will appear here after generation.",
            anchor="center",
            background="#1a1a1a",
            foreground="#888",
        )
        self.preview_label.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(
            parent, textvariable=self.status_var, relief="sunken",
            anchor="w", padding=(6, 2),
        ).grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _toggle_directional(self):
        enabled = self.use_directional.get()
        state = "normal" if enabled else "disabled"
        for lbl, ent in self._dir_rows:
            ent.configure(state=state)
        self.prompt_entry.configure(state="disabled" if enabled else "normal")

    def _browse_ckpt(self, var: tk.StringVar):
        current = var.get()
        if current and Path(current).exists():
            start_dir = os.path.dirname(current)
        else:
            start_dir = str(_script_dir())
        path = filedialog.askopenfilename(
            title="Select checkpoint",
            initialdir=start_dir,
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    def _on_generate(self):
        # Validate checkpoints
        for name, var in (("VAE", self.vae_var), ("U-Net", self.unet_var),
                          ("ControlNet", self.cn_var)):
            p = var.get().strip()
            if not p:
                messagebox.showerror("Missing checkpoint",
                                     f"Please select a {name} checkpoint.")
                return
            if not Path(p).exists():
                messagebox.showerror("File not found",
                                     f"{name} checkpoint not found:\n{p}")
                return

        seed_raw = self.seed_var.get().strip()
        try:
            seed = int(seed_raw) if seed_raw else None
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be an integer or blank.")
            return

        grid_size = int(self.grid_var.get())
        if grid_size % 2 == 0:
            messagebox.showerror("Invalid grid size",
                                 f"Grid size must be odd (got {grid_size}).")
            return

        try:
            overlap = int(self.overlap_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid overlap",
                                 "Overlap must be an integer in [0, 256).")
            return
        if not (0 <= overlap < 256):
            messagebox.showerror("Invalid overlap",
                                 f"Overlap must be in [0, 256) (got {overlap}).")
            return

        try:
            strength = float(self.strength_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid strength",
                                 "img2img strength must be in [0, 1].")
            return
        if not (0.0 <= strength <= 1.0):
            messagebox.showerror("Invalid strength",
                                 f"Strength must be in [0, 1] (got {strength:.2f}).")
            return

        use_dir = self.use_directional.get()
        if use_dir:
            if not any(v.get().strip() for v in
                       (self.north_var, self.south_var, self.east_var, self.west_var)):
                messagebox.showerror(
                    "No directional prompts",
                    "Directional mode is on but none of N/S/E/W have prompts.")
                return
        else:
            if not self.prompt_var.get().strip():
                messagebox.showerror("Missing prompt", "Please enter a prompt.")
                return

        self._set_log("")
        self.diag_var.set("Running…")
        self.save_png_btn.configure(state="disabled")
        self.save_npy_btn.configure(state="disabled")
        self.export_pkg_btn.configure(state="disabled")
        self._last_result = None

        params = {
            "vae_ckpt":          self.vae_var.get().strip(),
            "unet_ckpt":         self.unet_var.get().strip(),
            "controlnet_ckpt":   self.cn_var.get().strip(),
            "prompt":            self.prompt_var.get().strip(),
            "use_single_prompt": not use_dir,
            "prompt_north":      self.north_var.get().strip(),
            "prompt_south":      self.south_var.get().strip(),
            "prompt_east":       self.east_var.get().strip(),
            "prompt_west":       self.west_var.get().strip(),
            "grid_size":         grid_size,
            "cfg_scale":         float(self.cfg_scale_var.get()),
            "num_steps":         int(self.steps_var.get()),
            "seed":              seed,
            "img2img_strength":  strength,
            "overlap":           overlap,
        }
        self.worker.run_generation(params)

    def _on_save_png(self):
        if not self._last_result:
            return
        path = filedialog.asksaveasfilename(
            title="Save PNG",
            initialdir=self.cfg.get("last_output_dir", str(Path.home())),
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if path:
            self._last_result["image"].save(path)
            self.cfg["last_output_dir"] = os.path.dirname(path)
            self.status_var.set(f"Saved: {path}")

    def _on_save_npy(self):
        if not self._last_result:
            return
        path = filedialog.asksaveasfilename(
            title="Save heightmap as .npy",
            initialdir=self.cfg.get("last_output_dir", str(Path.home())),
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy")],
        )
        if path:
            np.save(path, self._last_result["stitched_raw"])
            self.cfg["last_output_dir"] = os.path.dirname(path)
            self.status_var.set(f"Saved: {path}")

    def _on_export_package(self):
        if not self._last_result:
            return
        folder = filedialog.askdirectory(
            title="Choose a folder to write terrain_package/ into",
            initialdir=self.cfg.get("last_output_dir", str(Path.home())),
            mustexist=True,
        )
        if not folder:
            return

        out_dir = os.path.join(folder, "terrain_package")
        try:
            from terrain_package import build_terrain_package
            meta = build_terrain_package(
                stitched_raw=self._last_result["stitched_raw"],
                prompt=self._last_result.get("prompt"),
                directional=self._last_result.get("directional") or {},
                output_dir=out_dir,
                seed=self._last_result.get("seed"),
                playable_mask=self._last_result.get("macro_params", {}).get("_playable_mask"),
                feature_masks=self._last_result.get("macro_params", {}).get("_feature_masks"),
            )
        except Exception as e:
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")
            self._append_log("Export ERROR:\n" + traceback.format_exc())
            return

        self.cfg["last_output_dir"] = folder
        self.status_var.set(f"Exported: {out_dir}")
        self._append_log(
            f"● Package written to {out_dir}\n"
            f"    trees:  {meta['counts']['trees']}\n"
            f"    rocks:  {meta['counts']['rocks']}\n"
            f"    plants: {meta['counts']['plants']}"
        )

    # ── Queue polling ────────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.worker.q.get_nowait()
                self._handle_msg(kind, payload)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _handle_msg(self, kind, payload):
        if kind == "log":
            self._append_log(payload)
        elif kind == "status":
            self.status_var.set(payload)
            self._append_log(f"● {payload}")
        elif kind == "preview":
            self._show_preview(payload)
        elif kind == "diag":
            d = payload
            self.diag_var.set(
                f"seams evaluated : {d['num_seams']}\n"
                f"mean L1         : {d['mean_l1']:.2f}\n"
                f"median L1       : {d['median_l1']:.2f}\n"
                f"max L1          : {d['max_l1']:.2f}\n"
                f"horiz-seam mean : {d['horiz_mean_l1']:.2f}\n"
                f"vert-seam mean  : {d['vert_mean_l1']:.2f}"
            )
        elif kind == "done":
            self._last_result = payload
            self.save_png_btn.configure(state="normal")
            self.save_npy_btn.configure(state="normal")
            self.export_pkg_btn.configure(state="normal")
            self.status_var.set("Done.")
        elif kind == "error":
            messagebox.showerror("Generation failed", payload)
            self._append_log("ERROR:\n" + payload)
            self.status_var.set("Error — see log.")
        elif kind == "busy":
            self.generate_btn.configure(
                state="disabled" if payload else "normal",
                text="Generating…" if payload else "Generate",
            )

    def _show_preview(self, pil_image):
        from PIL import ImageTk
        w = self.preview_label.winfo_width()
        h = self.preview_label.winfo_height()
        if w < 40 or h < 40:
            w, h = 700, 700
        img = pil_image.copy()
        img.thumbnail((w, h))
        self._preview_tk_image = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self._preview_tk_image, text="")

    def _append_log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        if text:
            self.log_text.insert("end", text)
        self.log_text.configure(state="disabled")

    def _report_autodetect(self):
        base = _script_dir()
        self._append_log(f"Scanning for checkpoints in: {base}")
        found_any = False
        for label, var in (("VAE", self.vae_var), ("U-Net", self.unet_var),
                           ("ControlNet", self.cn_var)):
            path = var.get()
            if path:
                try:
                    rel = Path(path).resolve().relative_to(base)
                    shown = str(rel)
                except ValueError:
                    shown = path
                self._append_log(f"  ✓ {label:10s} → {shown}")
                found_any = True
            else:
                self._append_log(f"  ✗ {label:10s} not found (pick one with the '…' button)")
        if found_any:
            self.status_var.set("Checkpoints auto-detected. Ready.")
        else:
            self.status_var.set("No checkpoints found — set paths manually.")

    def _on_close(self):
        self.cfg.update({
            "vae_ckpt":          self.vae_var.get().strip(),
            "unet_ckpt":         self.unet_var.get().strip(),
            "controlnet_ckpt":   self.cn_var.get().strip(),
            "prompt":            self.prompt_var.get().strip(),
            "grid_size":         int(self.grid_var.get()),
            "cfg_scale":         float(self.cfg_scale_var.get()),
            "num_steps":         int(self.steps_var.get()),
            "img2img_strength":  float(self.strength_var.get()),
            "overlap":           int(self.overlap_var.get()),
        })
        save_config(self.cfg)
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    TerrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
