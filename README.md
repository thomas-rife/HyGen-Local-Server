# HyGen Local Server

A local, prompt-driven terrain generation service for the HyGen
Hytale mod. It builds a procedural macro heightmap, refines it with a
text-conditioned diffusion model, and exports a package that the game server
downloads and places into a new world.

The HTTP service runs separately from Hytale. No application API key is required.
Prompt parsing is local; the T5-small tokenizer and text encoder are downloaded
on first use if they are not already cached.

## Full-Map Benchmarks

These are user-measured **end-to-end results for a full 640 x 640 map**. The MPS result uses GPU-accelerated
inference while procedural preprocessing and package export remain on the CPU.

| Hardware                | Inference backend | End-to-end time | Speedup vs. M5 Max CPU |
| ----------------------- | ----------------- | --------------- | ---------------------- |
| Apple M5 Max            | CPU               | 121 seconds     | 1.0x                   |
| Apple M5 Max            | MPS               | 18 seconds      | 6.7x                   |
| NVIDIA GeForce RTX 3080 | CUDA              | 34 seconds      | 3.6x                   |

The MPS run was 103 seconds faster, approximately an 85% reduction in total time.
The RTX 3080 also recorded **21 seconds for generation only**, separate from its
34-second end-to-end result. The generation-only time has a different timing
boundary and should not be compared directly with the end-to-end rows above.

### Model Size

| Terrain model | Parameters |
| ------------- | ---------- |
| VAE           | 3.8M       |
| U-Net         | 55.9M      |
| ControlNet    | 31.6M      |
| **Total**     | **91.3M**  |

Counts are rounded and cover the three terrain models only. T5-small is a separate
text encoder and is not included in that total.

### Benchmark World

| Setting                    | Recorded value                            |
| -------------------------- | ----------------------------------------- |
| World size                 | 640 x 640 heightmap cells (409,600 total) |
| Primitive / base shape     | `rolling` / `rolling_hills`               |
| Prompt parser route        | `fallback`                                |
| Seed                       | `100241644`                               |
| img2img strength           | `0.40`                                    |
| Constraint repair strength | `0.45`                                    |
| Playable center            | `(512, 214)`                              |
| Playable target height     | `66.00`                                   |
| Raw height range           | `46.47` to `77.60`                        |
| Normalized height range    | `-0.650` to `-0.430`                      |
| Macro validation           | Passed, zero errors and zero warnings     |
| Water / rivers             | None                                      |

Recorded generation output:

```text
[macro-validation]: ok=True errors=0 warnings=0 height_range=31.137 playable_cells=548.000 playable_mean_slope=0.078 water_fraction=0.000
[pipeline] primitive=rolling     modifiers=-                         ocean=-      rivers=0  world=640px  via=fallback base_shape=rolling_hills playable=(512,214) target=66.00 protect=1.00 water_cells=0 river_cells=0 feature_water=0 feature_playable=758 seed=100241644 img2img=0.40 repair=0.45  raw=(46.47,77.60) norm=(-0.650,-0.430)
```

### RTX 3080 Results

| Measurement                      | Result  |
| -------------------------------- | ------- |
| End-to-end full-map time         | 34 seconds |
| Generation-only time            | 21 seconds |
| CPU model / GPU VRAM capacity    | Pending |
| Python / PyTorch / CUDA versions | Pending |
| Sampling steps / CFG scale       | Pending |
| Cold or warm model cache         | Pending |
| Number of runs / timing method   | Pending |
| Validation result                | Pending |

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
Hytale world/block placement is handled by HyGen, not this server.

## Repository Layout

```text
HyGen-Local-Server/
  models/                     Git LFS terrain checkpoints
    vae_ema_final.pt
    unet_ema_final.pt
    controlnet_ema_final.pt
  hygen/                      Python implementation package
    checkpoints.py            Shared repository-relative model discovery
    terrain_server.py         HTTP service and pipeline orchestration
    terrain_gui.py            Optional desktop GUI
    generate_terrain.py       Legacy CLI, model loading, and text encoding
    generate_terrain_img2img.py
    terrain_cldm.py           Neural model definitions
    terrain_dataloader.py     Height normalization and data utilities
    terrain_macro.py          Procedural macro generation
    macro_prompt_parser.py
    terrain_scene_adapter.py
    terrain_validation.py
    terrain_package.py        Java-facing package export
  tests/                      Regression tests (no model weights needed)
  terrain_server.py           Backward-compatible server launcher
  terrain_gui.py              Backward-compatible GUI launcher
  generate_terrain.py         Backward-compatible legacy CLI launcher
  requirements.txt
  README.md
```

Use `outputs/` for manually exported previews or packages; it is ignored by Git.
Server-generated packages still default to the system temporary directory.
The root launch commands are unchanged. Checkpoints resolve relative to this
repository, not your current working directory.

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
models/vae_ema_final.pt
models/unet_ema_final.pt
models/controlnet_ema_final.pt
```

The server and GUI discover checkpoints in the repository's `models/` directory.
For another checkpoint directory, run
`python terrain_server.py --models_dir /path/to/checkpoints`. The legacy CLI
also defaults to `models/` and accepts explicit `--vae_ckpt`, `--unet_ckpt`, and
`--controlnet_ckpt` paths. Load only trusted checkpoints; the loader uses PyTorch
deserialization with `weights_only=False`.

### Windows / NVIDIA

Create a new environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build using the command from the
[official installation selector](https://pytorch.org/get-started/locally/)
for your GPU/driver, then install the remaining dependencies using the same
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe terrain_server.py --device cuda
```

`CUDA: True` confirms that PyTorch can see a CUDA device.

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

| Endpoint                           | Purpose                                                    |
| ---------------------------------- | ---------------------------------------------------------- |
| `GET /health`                      | HTTP liveness check; does not load or test models          |
| `POST /generate`                   | Generate a package and return its job ID and download URLs |
| `GET /package/{job_id}/{filename}` | Download a generated package file                          |
| `DELETE /package/{job_id}`         | Remove that job's package files                            |

Packages are written to `terrain_packages` inside the system temporary directory
unless `--storage_dir` overrides it. Other options include `--host`, `--port`, and
`--framework auto|fastapi|flask`; see `python terrain_server.py --help`.

Keep the default loopback binding for local use. The API has no authentication;
do not expose it publicly without adding appropriate access controls.

### HyGen Connection

Run this service alongside the Hytale server. In HyGen's
`run/universe/battleheart-ai-terrain.json`, set `pythonEndpoint` to
`http://localhost:8080` when both servers run on the same computer. The Java
plugin requests a package, downloads it, places terrain/water/decorations, and
prepares the generated world for play.

Cloning this repository does not install Hytale or copy HyGen's saved
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
  "img2img_strength": 0.4,
  "repair_strength": 0.45,
  "num_steps": 30,
  "cfg_scale": 5.0
}
```

For 256-cell chunks, a 3 x 3 grid with 64-cell overlap produces a 640 x 640 map:
`256 + (3 - 1) * (256 - 64) = 640`.

## Development

Run the regression tests from the repository root:

```sh
python -B -m unittest discover -s tests -v
```

Tests use temporary files and mocked neural models; they do not start the HTTP
service, download T5, or run full terrain inference. Python implementations can
also be launched as modules, for example `python -m hygen.terrain_server`.

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
