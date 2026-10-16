"""YAML config loader for cfDNA CNV caller.

Config 경로 우선순위:
    1) CFDNA_CNV_CONFIG 환경변수
    2) 현재 작업 디렉토리의 config.yaml
    3) repo 루트의 config.yaml (이 파일 기준 ../../config.yaml)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _candidate_paths() -> list[Path]:
    env_path = os.environ.get("CFDNA_CNV_CONFIG")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path.cwd() / "config.yaml")
    candidates.append(_REPO_ROOT / "config.yaml")
    return candidates


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config. Raise FileNotFoundError if no candidate exists."""
    if path is not None:
        cfg_path = Path(path).expanduser()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
    else:
        cfg_path = next((p for p in _candidate_paths() if p.is_file()), None)
        if cfg_path is None:
            tried = "\n  ".join(str(p) for p in _candidate_paths())
            raise FileNotFoundError(
                "No config.yaml found. Set CFDNA_CNV_CONFIG or create "
                "config.yaml (cp config.example.yaml config.yaml).\n"
                f"Tried:\n  {tried}"
            )

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_path = cfg_path.resolve()
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    path_keys = {"work_dir", "qdnaseq_matrix", "pon_reference", "meta_xlsx",
                 "out_dir", "extract_cnr_script", "ref_qdnaseq_dir"}
    for key in path_keys.intersection(cfg.get("paths", {})):
        value = cfg["paths"][key]
        if value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = cfg_path.parent / candidate
            cfg["paths"][key] = str(candidate.resolve())
    cfg["_config_path"] = str(cfg_path)
    return cfg


def get_path(cfg: Dict[str, Any], key: str, must_exist: bool = True) -> Path:
    """Helper to fetch a path from cfg['paths'][key] with existence check."""
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"paths.{key} not in config ({cfg.get('_config_path')})")
    p = Path(paths[key]).expanduser()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"paths.{key} = {p} does not exist")
    return p
