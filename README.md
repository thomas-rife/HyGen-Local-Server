# HyGen Local Server

A local, prompt-driven terrain generation service for the BattleHeartClone
Hytale plugin. It builds a procedural macro heightmap, refines it with a
text-conditioned diffusion model, and exports a package that the game server
downloads and places into a new world.

The HTTP service runs separately from Hytale. No application API key is required.
Prompt parsing is local; the T5-small tokenizer and text encoder are downloaded
on first use if they are not already cached.

## Full-Map Benchmarks

These are user-measured **end-to-end results for a full 640 x 640 map**, not
isolated neural-network inference timings. The MPS result uses GPU-accelerated
inference while procedural preprocessing and package export remain on the CPU.

| Hardware | Inference backend | End-to-end time | Speedup vs. M5 Max CPU |
| --- | --- | --- | --- |
| Apple M5 Max | CPU | 121 seconds | 1.0x |
| Apple M5 Max | MPS | 18 seconds | 6.7x |
| NVIDIA GeForce RTX 3080 | CUDA | Pending | Pending |

The MPS run was 103 seconds faster, approximately an 85% reduction in total time.
These measurements were supplied by the project author and have not been
independently reproduced here. Cold/warm model-loading state, software versions,
and sampling-step count were not recorded with the supplied results.

### Model Size

| Terrain model | Parameters |
| --- | --- |
| VAE | 3.8M |
| U-Net | 55.9M |
| ControlNet | 31.6M |
| **Total** | **91.3M** |

Counts are rounded and cover the three terrain models only. T5-small is a separate
text encoder and is not included in that total.

### Benchmark World

| Setting | Recorded value |
| --- | --- |
| World size | 640 x 640 heightmap cells (409,600 total) |
| Primitive / base shape | `rolling` / `rolling_hills` |
| Prompt parser route | `fallback` |
| Seed | `100241644` |
| img2img strength | `0.40` |
| Constraint repair strength | `0.45` |
| Playable center | `(512, 214)` |
| Playable target height | `66.00` |
| Raw height range | `46.47` to `77.60` |
| Normalized height range | `-0.650` to `-0.430` |
| Macro validation | Passed, zero errors and zero warnings |
| Water / rivers | None |

Recorded generation output:

```text
[macro-validation]: ok=True errors=0 warnings=0 height_range=31.137 playable_cells=548.000 playable_mean_slope=0.078 water_fraction=0.000
[pipeline] primitive=rolling     modifiers=-                         ocean=-      rivers=0  world=640px  via=fallback base_shape=rolling_hills playable=(512,214) target=66.00 protect=1.00 water_cells=0 river_cells=0 feature_water=0 feature_playable=758 seed=100241644 img2img=0.40 repair=0.45  raw=(46.47,77.60) norm=(-0.650,-0.430)
```

### RTX 3080 Results To Add

| Measurement | Result |
| --- | --- |
| End-to-end full-map time | Pending |
| CPU model / GPU VRAM capacity | Pending |
| Python / PyTorch / CUDA versions | Pending |
| Sampling steps / CFG scale | Pending |
| Cold or warm model cache | Pending |
| Number of runs / timing method | Pending |
| Validation result | Pending |

For a useful comparison, keep the prompt, seed, checkpoints, map dimensions,
sampling settings, and postprocessing settings the same. Record whether model
loading and Hytale download/placement are included in the timing boundary.

## Generation Pipeline

1. Parse the prompt into procedural macro parameters, without an LLM call.
2. Build and validate the global macro heightmap and playable area.
3. Encode text with T5-small and refine overlapping terrain chunks using the
   VAE, U-Net, and ControlNet img2img pipeline.
4. Crossfade the chunks, denormalize heights, and repair scene constraints.
5. Validate the result and export heights, materials, water, decorations, and
   a preview image.

The neural models use the selected PyTorch device. Macro construction,
validation, crossfade stitching, and package construction use CPU processing.
Hytale world/block placement is handled by BattleHeartClone, not this server.

## Setup

Requirements: Git, [Git LFS](https://git-lfs.com/), and a Python version supported
by your chosen [PyTorch build](https://pytorch.org/get-started/locally/).
Run all commands below from the cloned repository root.

### Model Checkpoints

Install Git LFS, then fetch the checkpoint contents:

```sh
git lfs install
git lfs pull
```

The following files must contain actual model weights, not Git LFS pointer text:

```text
vae_ema_final.pt
unet_ema_final.pt
controlnet_ema_final.pt
```

The server discovers checkpoints under its working directory. Load only trusted
checkpoints; the loader uses PyTorch deserialization with `weights_only=False`.

### Windows / NVIDIA

Create a new environment; do not copy a macOS virtual environment to Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build using the command from the
[official installation selector](https://pytorch.org/get-started/locally/)
for your GPU/driver. Use `.\.venv\Scripts\python.exe -m pip` in place of the
selector's `pip` or `pip3` command to target this environment. Then install the
remaining dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe terrain_server.py --device cuda
```

`CUDA: True` confirms that PyTorch can see a CUDA device. An explicit
`--device cuda` does not silently fall back to CPU if CUDA is unavailable.

### macOS / Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python terrain_server.py
```

The HTTP server defaults to **CUDA when available, otherwise CPU**. On Apple
Silicon, select MPS explicitly to use the GPU:

```sh
python terrain_server.py --device mps
```

To force CPU on any platform:

```sh
python terrain_server.py --device cpu
```

Dependencies are currently unpinned. Record installed versions when collecting
benchmarks so results can be compared meaningfully.

## Running And Integration

The server listens on `http://127.0.0.1:8080` by default. FastAPI/uvicorn is
preferred, with Flask as a fallback. Models are loaded lazily on the first
generation request, cached between requests, and generation jobs are serialized.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | HTTP liveness check; does not load or test models |
| `POST /generate` | Generate a package and return its job ID and download URLs |
| `GET /package/{job_id}/{filename}` | Download a generated package file |
| `DELETE /package/{job_id}` | Remove that job's package files |

Packages are written to `terrain_packages` inside the system temporary directory
unless `--storage_dir` overrides it. Other options include `--host`, `--port`, and
`--framework auto|fastapi|flask`; see `python terrain_server.py --help`.

Keep the default loopback binding for local use. The API has no authentication;
do not expose it publicly without adding appropriate access controls.

### BattleHeartClone Connection

Run this service alongside the Hytale server. In BattleHeartClone's
`run/universe/battleheart-ai-terrain.json`, set `pythonEndpoint` to
`http://localhost:8080` when both servers run on the same computer. The Java
plugin requests a package, downloads it, places terrain/water/decorations, and
prepares the generated world for play.

Cloning this repository does not install Hytale or copy BattleHeartClone's saved
worlds. Those belong to the separate game-server setup.

### Example Generation Request

This example uses the recorded seed, img2img strength, and repair strength.
The prompt, step count, and CFG scale below are example values, not a complete
record of the benchmark invocation.

```json
{
  "prompt": "rolling hills",
  "seed": 100241644,
  "grid_size": 3,
  "overlap": 64,
  "img2img_strength": 0.40,
  "repair_strength": 0.45,
  "num_steps": 30,
  "cfg_scale": 5.0
}
```

For 256-cell chunks, a 3 x 3 grid with 64-cell overlap produces a 640 x 640 map:
`256 + (3 - 1) * (256 - 64) = 640`.

## Timing Logs

Each `/generate` request prints a UTC start timestamp and elapsed seconds to
the Python server console. For example:

```text
[generation <job-id>] started at <UTC timestamp>
[generation <job-id>] completed in <elapsed>s
```

Failed requests print `failed after <elapsed>s` and preserve the original error.
The server timer covers the request through package generation, including model
loading on a cold request and any wait for another generation. It does **not**
include subsequent HTTP downloads, Hytale terrain placement, or teleportation.
Restart the Python server after source changes to load the new code.

## Troubleshooting

- **CUDA unavailable:** check the active Python environment, NVIDIA driver, and
  CUDA-enabled PyTorch installation. The default server uses CPU in this case.
- **Checkpoint loading fails:** run `git lfs pull` and verify all three weights
  were downloaded completely.
- **First request is slower:** it includes model loading and may include the
  first T5-small download; later requests reuse the cached models.
- **Game cannot reach the service:** confirm both servers are running and the
  configured endpoint matches the Python server's host and port.
- **MPS benchmark reproduction:** pass `--device mps`; the server no longer
  chooses MPS automatically.
