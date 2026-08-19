from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def current_git_commit(project_root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root) if project_root else None,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_metadata(config_path: str | Path, config: dict, project_root: str | Path | None = None, model_version: str | None = None) -> dict:
    return {
        "config_path": str(Path(config_path)),
        "dataset": config.get("dataset", {}).get("name"),
        "random_seed": config.get("experiment", {}).get("random_seed"),
        "model_version": model_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(project_root),
    }
