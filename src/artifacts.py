"""Download serving artifacts from the configured Hugging Face repository."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "briankm2004/housing-regression-serving-artifacts"


def ensure_artifact(relative_path: str, project_root: Path) -> Path:
    """Return a local artifact path, downloading it when it is not present."""
    local_path = project_root / relative_path
    if local_path.exists():
        return local_path

    repo_id = os.getenv("HF_REPO_ID", DEFAULT_REPO_ID)
    token = os.getenv("HF_TOKEN")
    hf_hub_download(
        repo_id=repo_id,
        filename=relative_path.replace("\\", "/"),
        token=token,
        local_dir=str(project_root),
    )
    if not local_path.exists():
        raise FileNotFoundError(
            f"Hugging Face download completed but the artifact is missing: {local_path}"
        )
    return local_path
