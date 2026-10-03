"""Repository-relative checkpoint paths shared by the server, GUI, and CLI."""

from pathlib import Path
from typing import Dict, Optional, Union


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
MODEL_FILENAMES = {
    "vae": "vae_ema_final.pt",
    "unet": "unet_ema_final.pt",
    "controlnet": "controlnet_ema_final.pt",
}


def find_model_ckpts(
    search_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[str]]:
    """Prefer standard filenames, then deterministically match .pt/.pth files."""
    root = Path(search_dir).expanduser().resolve() if search_dir is not None else DEFAULT_MODELS_DIR
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
    )
    found = {}
    for role, filename in MODEL_FILENAMES.items():
        exact = root / filename
        if exact.is_file():
            found[role] = str(exact)
            continue
        match = next(
            (path for path in candidates
             if role in path.name.lower()
             and not (role == "unet" and "controlnet" in path.name.lower())),
            None,
        )
        found[role] = str(match) if match is not None else None
    return found
