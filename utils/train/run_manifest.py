# utils/train/run_manifest.py
from __future__ import annotations
import os, sys, platform, subprocess, json, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

def _git(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return out
    except Exception:
        return None

def get_git_info() -> Dict[str, Any]:
    head = _git(["git", "rev-parse", "HEAD"])
    short = _git(["git", "rev-parse", "--short", "HEAD"])
    status = _git(["git", "status", "--porcelain"])
    is_dirty = bool(status)
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {"commit": head, "short": short, "branch": branch, "dirty": is_dirty}

def collect_env_info() -> Dict[str, Any]:
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "numpy": _v("numpy"),
            "pandas": _v("pandas"),
            "xgboost": _v("xgboost"),
            "scikit_learn": _v("sklearn"),
            "optuna": _v("optuna"),
            "pyarrow": _v("pyarrow"),
        },
    }

def quick_file_fingerprint(p: Path) -> Dict[str, Any]:
    """Avoid O(dataset) hashing by using size+mtime. Good enough for lineage."""
    try:
        st = p.stat()
        key = f"{p.name}:{int(st.st_size)}:{int(st.st_mtime)}".encode()
        digest = hashlib.sha256(key).hexdigest()
        return {"path": str(p), "size": int(st.st_size), "mtime": int(st.st_mtime), "sha256_quick": digest}
    except Exception:
        return {"path": str(p), "error": "unavailable"}

def make_run_id(run_tag: Optional[str]=None) -> str:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    gi = get_git_info()
    sh = gi.get("short") or "nogit"
    tag = (run_tag or "").strip().replace(" ", "-")
    return f"{now}_{sh}" + (f"_{tag}" if tag else "")

def _manifest_dir(trainer) -> Path:
    d = trainer.paths.model_dir / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d

def manifest_path(trainer) -> Path:
    return _manifest_dir(trainer) / f"{trainer.run_id}.yaml"

def write_manifest(trainer, stage: str, extra: Optional[Dict[str, Any]]=None):
    """
    stage: 'start' or 'end'. On 'start' we create, on 'end' we update.
    """
    path = manifest_path(trainer)
    data: Dict[str, Any] = {}
    if stage == "start":
        gi = get_git_info()
        env = collect_env_info()
        fm = Path(trainer.paths.feature_matrix_path)
        data = {
            "run_id": trainer.run_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "git": gi,
            "env": env,
            "config_path": str(trainer.config_path),
            "feature_matrix": quick_file_fingerprint(fm),
            "problems": [p["name"] for p in trainer.problems],
            "models": list(trainer.cfg.models_to_train),
            "production_mode": bool(trainer.cfg.production_mode),
            "versioning_mode": getattr(trainer, "versioning_mode", "run_id"),
            "base_seed": getattr(trainer, "base_seed", None),
            "run_tag": getattr(trainer, "run_tag", None),
        }
        if extra:
            data.update(extra)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return
    # update/append
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    data["ended_at"] = datetime.now().isoformat(timespec="seconds")
    if extra:
        data.setdefault("results", {}).update(extra)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
