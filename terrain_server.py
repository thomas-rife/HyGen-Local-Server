"""
terrain_server.py — Local HTTP service that wraps the terrain pipeline.

Updated for the img2img pipeline:
    1. Parse the prompt locally (NO LLM) into macro parameters.
    2. Build the global macro heightmap from the parsed macro scene.
    3. Run generate_grid(...) with img2img refinement on top of the macro.
    4. Crossfade-stitch the overlapping chunks.
    5. Build the Java-facing terrain_package as before.

The Java mod talks to this server. On POST /generate it returns a job ID
plus URLs Java can GET.

New JSON request fields (all optional with sensible defaults):
    "img2img_strength" : float in [0,1], default 0.4
    "overlap"          : int   in [0,128], default 64

This prefers FastAPI + uvicorn if both are available. Falls back to Flask.

Run:
    python terrain_server.py
"""

import argparse
import os
import secrets
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional


# Lazy heavy imports — keep the server start-up snappy.


def resolve_generation_seed(seed: Optional[int]) -> int:
    """Pick the concrete seed for this run.

    Explicit seeds stay reproducible. Missing seeds become fresh random seeds.
    """
    if seed is not None:
        return int(seed) & 0xFFFFFFFF
    return int(secrets.randbelow(2 ** 32))


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_model_ckpts(search_dir: str = "."):
    """Walk a directory tree and pick the first VAE/U-Net/ControlNet found."""
    search_dir = Path(search_dir).resolve()
    found = {"vae": None, "unet": None, "controlnet": None}

    for p in search_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()

        # Order matters: check controlnet before unet so a "controlnet_unet*"
        # filename is classified correctly.
        if not found["controlnet"] and "controlnet" in name:
            found["controlnet"] = str(p)
        elif not found["unet"] and "unet" in name and "controlnet" not in name:
            found["unet"] = str(p)
        elif not found["vae"] and "vae" in name:
            found["vae"] = str(p)

        if all(found.values()):
            break

    return found


# ══════════════════════════════════════════════════════════════════════════════
# PACKAGE STORE ABSTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class _PackageStore:
    """Local filesystem store. Replace with a GCS-backed subclass later."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def allocate_dir(self, job_id: str) -> str:
        d = os.path.join(self.root, job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def path_for(self, job_id: str, filename: str) -> str:
        # Protect against path traversal — only allow basenames.
        safe = os.path.basename(filename)
        return os.path.join(self.root, job_id, safe)

    def delete(self, job_id: str) -> bool:
        d = os.path.join(self.root, job_id)
        if not os.path.isdir(d):
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE WRAPPER  (model caching + one-shot generate → package)
# ══════════════════════════════════════════════════════════════════════════════

class _Pipeline:
    """Loads models once and runs generation jobs synchronously.

    Thread-safe via a single mutex: only one generation at a time, since the
    underlying CUDA context is shared.
    """

    def __init__(self, vae_ckpt: str, unet_ckpt: str, controlnet_ckpt: str,
                 device: Optional[str] = None):
        self.vae_ckpt = vae_ckpt
        self.unet_ckpt = unet_ckpt
        self.controlnet_ckpt = controlnet_ckpt
        self.device = device
        self._models = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._models is not None:
            return

        import torch
        # We import the *legacy* generate_terrain only for load_models +
        # T5TextEncoder + CosineNoiseSchedule. The actual sampling /
        # stitching path goes through generate_terrain_img2img.
        import generate_terrain as gt_legacy

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[pipeline] loading models on {device}...", flush=True)
        vae, unet, controlnet = gt_legacy.load_models(
            self.vae_ckpt, self.unet_ckpt, self.controlnet_ckpt, device,
        )
        text_encoder = gt_legacy.T5TextEncoder(device=device)
        schedule = gt_legacy.CosineNoiseSchedule(num_timesteps=1000)
        self._models = (vae, unet, controlnet, text_encoder, schedule, device)
        print("[pipeline] ready.", flush=True)

    def run(
        self, *,
        prompt: Optional[str], directional: Dict[str, Optional[str]],
        grid_size: int, sea_level: int, origin_x: int, origin_z: int,
        base_y: int, cfg_scale: float, num_steps: int,
        seed: Optional[int],
        smooth: bool, smooth_strength: float, feature_smoothing: bool,
        decorations: bool,
        img2img_strength: float, overlap: int, repair_strength: float,
        out_dir: str,
    ) -> dict:
        if grid_size % 2 == 0:
            raise ValueError(f"grid_size must be odd (got {grid_size})")
        if not (0.0 <= img2img_strength <= 1.0):
            raise ValueError(
                f"img2img_strength must be in [0, 1] (got {img2img_strength})"
            )
        if not (0 <= overlap < 256):
            raise ValueError(f"overlap must be in [0, 256) (got {overlap})")

        seed = resolve_generation_seed(seed)

        with self._lock:
            self._ensure_loaded()

            # Imports inside the lock so the lazy load above can finish first.
            import generate_terrain as gt_legacy             # for prompt-pick + spiral helpers
            import generate_terrain_img2img as gt
            import terrain_macro
            from terrain_package import build_terrain_package
            from terrain_dataloader import denormalize_height
            from macro_prompt_parser import (
                parse_prompt_to_macro_params, summarize,
            )
            from terrain_scene_adapter import (
                build_normalized_macro_world,
                apply_constraint_repair,
                summarize_composition,
            )
            from terrain_validation import (
                validate_macro,
                validate_final_package_inputs,
                format_validation_lines,
            )
            print(f"[pipeline] using seed={seed}", flush=True)

            vae, unet, controlnet, text_encoder, schedule, device = self._models

            # ── 1. Parse prompt → macro params ─────────────────────────────
            # ── 1. Parse prompt → macro params ─────────────────────────────
            # ── 1. Parse prompt → macro params ─────────────────────────────
            macro_params = parse_prompt_to_macro_params(
                prompt,
                grid_size=grid_size,
                overlap=overlap,
                seed=seed,
                directional=directional,
            )

            # Adapter: parser schema → terrain_macro schema.
            # The parser returns "primitive"; terrain_macro expects "base_shape".
            



            # ── 2. Build the global macro heightmap ────────────────────────

            macro_world, macro_params_adapted, macro_stats = build_normalized_macro_world(
                terrain_macro,
                macro_params,
                base_y=base_y,
                sea_level=sea_level,
            )
            macro_raw_for_validation = macro_params_adapted.get("_macro_raw_with_playable")
            if macro_raw_for_validation is not None:
                macro_report = validate_macro(
                    macro_raw_for_validation,
                    macro_params_adapted,
                    sea_level=sea_level,
                )
                for line in format_validation_lines(macro_report, "[macro-validation]"):
                    print(line, flush=True)

                if macro_report.errors:
                    raise ValueError("; ".join(macro_report.errors))
            print(
                f"[pipeline] {summarize(macro_params)} "
                f"base_shape={macro_params_adapted.get('base_shape')} "
                f"playable=({macro_stats['playable_center_r']:.0f},{macro_stats['playable_center_c']:.0f}) "
                f"target={macro_stats['playable_target']:.2f} "
                f"protect={macro_stats.get('protect_max', 0.0):.2f} "
                f"water_cells={macro_stats.get('water_mask_cells', 0.0):.0f} "
                f"river_cells={macro_stats.get('river_mask_cells', 0.0):.0f} "
                f"feature_water={macro_stats.get('feature_water_cells', 0.0):.0f} "
                f"feature_playable={macro_stats.get('feature_playable_cells', 0.0):.0f} "
                f"seed={seed} "
                f"img2img={img2img_strength:.2f} "
                f"repair={repair_strength:.2f} "
                f"{summarize_composition(macro_params_adapted)} "
                f"raw=({macro_stats['raw_min']:.2f},{macro_stats['raw_max']:.2f}) "
                f"norm=({macro_stats['norm_min']:.3f},{macro_stats['norm_max']:.3f})",
                flush=True,
            )

            # ── 3. img2img refinement over the macro ───────────────────────
            t_start = int(round(img2img_strength * (schedule.T - 1)))

            chunks, _prompts_used, origins_y, origins_x = gt.generate_grid(
                grid_size=grid_size,
                macro_world=macro_world,
                vae=vae, unet=unet, controlnet=controlnet,
                schedule=schedule, text_encoder=text_encoder,
                device=device,
                cfg_scale=cfg_scale,
                num_steps=num_steps,
                single_prompt=prompt,
                directional=directional,
                seed=seed,
                overlap=overlap,
                start_timestep=t_start,
                pick_prompt_for_chunk_fn=gt_legacy.pick_prompt_for_chunk,
                spiral_order_fn=gt_legacy.spiral_order,
            )

            # ── 4. Crossfade stitch ────────────────────────────────────────
            stitched_norm = gt.stitch_chunks_crossfade(
                chunks, grid_size,
                origins_y=origins_y, origins_x=origins_x,
                overlap=overlap,
            )

            # Denormalize to raw Y-levels using the legacy helper so we
            # match the dataloader's normalization conventions exactly.
            from terrain_dataloader import denormalize_height
            stitched_raw = denormalize_height(stitched_norm)

            macro_raw_with_playable = macro_params_adapted.get("_macro_raw_with_playable")

            if macro_raw_with_playable is not None:
                stitched_raw = apply_constraint_repair(
                    stitched_raw,
                    macro_raw_with_playable,
                    macro_params_adapted,
                    sea_level=sea_level,
                    strength=repair_strength,
                )

                playable_meta = macro_params_adapted.get("_playable_area") or {}
                print(
                    f"[pipeline] constraints repaired "
                    f"playable=({playable_meta.get('center_r')},"
                    f"{playable_meta.get('center_c')}) "
                    f"target={float(playable_meta.get('target_height', 0.0)):.2f}",
                    flush=True,
                )

            # ── 5. Build the Java package ──────────────────────────────────
            final_report = validate_final_package_inputs(
                stitched_raw,
                macro_params_adapted,
                sea_level=sea_level,
            )
            for line in format_validation_lines(final_report, "[final-validation]"):
                print(line, flush=True)

            if final_report.errors:
                raise ValueError("; ".join(final_report.errors))

            return build_terrain_package(
                stitched_raw=stitched_raw,
                prompt=prompt,
                directional=directional,
                output_dir=out_dir,
                origin_x=origin_x,
                origin_z=origin_z,
                base_y=base_y,
                smooth=smooth,
                add_materials=True,
                add_decorations=decorations,
                sea_level=sea_level,
                smooth_strength=smooth_strength,
                feature_smoothing=feature_smoothing,
                seed=seed,
                playable_mask=macro_params_adapted.get("_playable_mask"),
                feature_masks=macro_params_adapted.get("_feature_masks"),
            )


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST HANDLING  —  shared between FastAPI and Flask paths
# ══════════════════════════════════════════════════════════════════════════════

def _run_generate_job(pipeline: _Pipeline, store: _PackageStore,
                      payload: dict, base_url: str) -> dict:
    """Shared request handler. Accepts a dict payload, returns the response dict."""
    job_id = uuid.uuid4().hex[:12]
    out_dir = store.allocate_dir(job_id)

    directional = {
        "north": payload.get("prompt_north"),
        "south": payload.get("prompt_south"),
        "east":  payload.get("prompt_east"),
        "west":  payload.get("prompt_west"),
    }
    prompt = payload.get("prompt")
    if prompt is None and not any(directional.values()):
        raise ValueError(
            "Provide 'prompt' or at least one of "
            "'prompt_north'/'prompt_south'/'prompt_east'/'prompt_west'."
        )
    if prompt is not None and any(directional.values()):
        # Match the legacy behaviour: single prompt wins when both are given.
        directional = {k: None for k in directional}

    creative_mode = bool(payload.get("creative_mode", True))
    img2img_strength = float(payload.get("img2img_strength", 0.55 if creative_mode else 0.4))
    repair_strength = float(payload.get("repair_strength", 0.45 if creative_mode else 0.75))

    metadata = pipeline.run(
        prompt=prompt,
        directional=directional,
        grid_size=int(payload.get("grid_size", 3)),
        sea_level=int(payload.get("seaLevel", 63)),
        origin_x=int(payload.get("originX", 0)),
        origin_z=int(payload.get("originZ", 0)),
        base_y=int(payload.get("baseY", 0)),
        cfg_scale=float(payload.get("cfg_scale", 5.0)),
        num_steps=int(payload.get("num_steps", 30)),    # default lowered: img2img needs fewer
        seed=payload.get("seed"),
        smooth=bool(payload.get("smooth", True)),
        smooth_strength=float(payload.get("smooth_strength", 0.3)),
        feature_smoothing=bool(payload.get("feature_smoothing", True)),
        decorations=bool(payload.get("decorations", True)),
        img2img_strength=img2img_strength,
        overlap=int(payload.get("overlap", 64)),
        repair_strength=repair_strength,
        out_dir=out_dir,
    )

    pkg_url = f"{base_url.rstrip('/')}/package/{job_id}"
    return {
        "jobId": job_id,
        "status": "done",
        "metadataUrl":    f"{pkg_url}/metadata.json",
        "heightmapUrl":   f"{pkg_url}/heightmap.bin.gz",
        "materialmapUrl": f"{pkg_url}/materialmap.bin.gz",
        "watermapUrl":    f"{pkg_url}/watermap.bin.gz",
        "waterheightUrl": f"{pkg_url}/waterheight.bin.gz",
        "decorationsUrl": f"{pkg_url}/decorations.json.gz",
        "seed":          metadata.get("seed"),
    }


_ALLOWED_FILES = {
    "metadata.json":        "application/json",
    "heightmap.bin.gz":     "application/gzip",
    "materialmap.bin.gz":   "application/gzip",
    "watermap.bin.gz":      "application/gzip",
    "waterheight.bin.gz":   "application/gzip",
    "decorations.json.gz":  "application/gzip",
    "preview.png":          "image/png",
}


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def _run_fastapi(pipeline: _Pipeline, store: _PackageStore,
                 host: str, port: int) -> None:
    from fastapi import FastAPI, HTTPException
    from starlette.requests import Request
    from fastapi.responses import FileResponse, JSONResponse
    import uvicorn

    app = FastAPI(title="Terrain Generation Service")

    def _base_url(request: Request) -> str:
        return f"{request.url.scheme}://{request.url.netloc}"

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(request: Request):
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Body must be JSON.")
        try:
            result = _run_generate_job(pipeline, store, payload, _base_url(request))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"{type(e).__name__}: {e}")
        return JSONResponse(result)

    @app.get("/package/{job_id}/{filename}")
    def get_package_file(job_id: str, filename: str):
        if filename not in _ALLOWED_FILES:
            raise HTTPException(status_code=404, detail="Unknown file.")
        path = store.path_for(job_id, filename)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Not found.")
        return FileResponse(path, media_type=_ALLOWED_FILES[filename],
                            filename=filename)

    @app.delete("/package/{job_id}")
    def delete_package(job_id: str):
        if store.delete(job_id):
            return {"jobId": job_id, "deleted": True}
        raise HTTPException(status_code=404, detail="Not found.")

    uvicorn.run(app, host=host, port=port, log_level="info")


# ══════════════════════════════════════════════════════════════════════════════
# FLASK FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _run_flask(pipeline: _Pipeline, store: _PackageStore,
               host: str, port: int) -> None:
    from flask import Flask, jsonify, request, send_file, abort

    app = Flask("terrain_server")

    def _base_url() -> str:
        return request.host_url.rstrip("/")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/generate")
    def generate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Body must be JSON."}), 400
        try:
            result = _run_generate_job(pipeline, store, payload, _base_url())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        return jsonify(result)

    @app.get("/package/<job_id>/<filename>")
    def get_package_file(job_id, filename):
        if filename not in _ALLOWED_FILES:
            abort(404)
        path = store.path_for(job_id, filename)
        if not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype=_ALLOWED_FILES[filename],
                         as_attachment=False, download_name=filename)

    @app.delete("/package/<job_id>")
    def delete_package(job_id):
        if store.delete(job_id):
            return jsonify({"jobId": job_id, "deleted": True})
        abort(404)

    app.run(host=host, port=port, threaded=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Local terrain generation HTTP service")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--storage_dir", type=str,
                   default=os.path.join(tempfile.gettempdir(), "terrain_packages"),
                   help="Where to write per-job package directories.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--framework", choices=("auto", "fastapi", "flask"),
                   default="auto",
                   help="Which HTTP framework to use. 'auto' prefers FastAPI.")
    args = p.parse_args()

    ckpts = find_model_ckpts(".")
    if not all(ckpts.values()):
        raise SystemExit("Couldn't find vae, unet, and controlnet checkpoints.")
    print("[models]", ckpts)

    store = _PackageStore(args.storage_dir)
    pipeline = _Pipeline(
        ckpts["vae"], ckpts["unet"], ckpts["controlnet"], device=args.device,
    )

    print(f"[server] storage_dir = {store.root}")
    print(f"[server] listening on http://{args.host}:{args.port}")

    if args.framework in ("auto", "fastapi"):
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
            _run_fastapi(pipeline, store, args.host, args.port)
            return
        except ImportError:
            if args.framework == "fastapi":
                print("FastAPI requested but not installed. "
                      "Install with: pip install fastapi uvicorn",
                      file=sys.stderr)
                sys.exit(1)
            print("[server] FastAPI not found; falling back to Flask.")

    try:
        import flask  # noqa: F401
    except ImportError:
        print("Neither FastAPI nor Flask is installed. "
              "Install one: pip install fastapi uvicorn  OR  pip install flask",
              file=sys.stderr)
        sys.exit(1)
    _run_flask(pipeline, store, args.host, args.port)


if __name__ == "__main__":
    main()
