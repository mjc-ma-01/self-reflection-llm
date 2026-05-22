from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"
SFT_DIR = DATASETS_DIR / "sft"
SFT_SOURCE_DIR = SFT_DIR / "source"
RL_DIR = DATASETS_DIR / "rl"

CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REQUIREMENTS_DIR = PROJECT_ROOT / "requirements"

SHARED_HF_HOME = Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "huggingface"))
SHARED_HF_CACHE = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", SHARED_HF_HOME / "hub"))
BACKUP_ROOT = Path(os.environ.get("SELF_REFLECTION_BACKUP_ROOT", PROJECT_ROOT / ".backups" / "self-reflection-source"))


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_runtime_dirs() -> None:
    for path in (OUTPUTS_DIR, RESULTS_DIR, CHECKPOINTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_hf_cache_env() -> None:
    os.environ.setdefault("HF_HOME", str(SHARED_HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(SHARED_HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(SHARED_HF_CACHE))
    os.environ.setdefault("HF_DATASETS_CACHE", str(SHARED_HF_HOME / "datasets"))


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.absolute()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.absolute()))
    except ValueError:
        return str(resolved)
