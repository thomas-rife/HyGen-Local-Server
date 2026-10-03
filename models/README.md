# Model Checkpoints

This directory contains the terrain models tracked with Git LFS:

- `vae_ema_final.pt`
- `unet_ema_final.pt`
- `controlnet_ema_final.pt`

After cloning, install Git LFS and run `git lfs pull` from the repository root.
The server and GUI discover these files automatically. T5-small is downloaded
through Transformers and uses its normal external cache, not this directory.

To use another checkpoint directory with the server, pass
`--models_dir /path/to/checkpoints`. Load only trusted checkpoint files.
