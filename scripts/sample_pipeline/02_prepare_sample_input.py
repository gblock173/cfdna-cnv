#!/usr/bin/env python3
"""Create a target-only matrix and derived YAML backed by compact PoN stats."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from caller.utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QDNAseq 1개와 depth별 PoN NPZ를 단일 샘플 calling 입력으로 준비합니다."
    )
    parser.add_argument("--aid", required=True)
    parser.add_argument("--qdnaseq", required=True, type=Path)
    parser.add_argument("--depth", required=True, choices=("100G", "20G", "WGS"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_yaml_dump(data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


def atomic_json_dump(data: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def validate_meta(config: dict, aid: str) -> None:
    meta_path = Path(config["paths"]["meta_xlsx"])
    sheet = config["paths"].get("meta_sheet", "전체")
    meta = pd.read_excel(meta_path, sheet_name=sheet, usecols=["AnalysisID"])
    if aid not in set(meta["AnalysisID"].dropna().astype(str)):
        raise SystemExit(
            f"{aid}가 meta XLSX에 없습니다. caller/profile용 메타데이터를 먼저 "
            f"추가하세요: {meta_path}"
        )


def load_signal(path: Path) -> tuple[pd.Series, str]:
    frame = pd.read_csv(path, sep="\t", dtype=str)
    metadata = {"feature", "chromosome", "start", "end"}
    missing = metadata.difference(frame.columns)
    if missing:
        raise SystemExit(f"QDNAseq 필수 컬럼 누락 {sorted(missing)}: {path}")
    value_columns = [column for column in frame.columns if column not in metadata]
    if len(value_columns) != 1:
        raise SystemExit(f"QDNAseq signal 컬럼이 1개여야 함: {value_columns}")
    if frame["feature"].duplicated().any():
        raise SystemExit("QDNAseq feature가 중복되어 있습니다.")

    value_column = value_columns[0]
    signal = pd.to_numeric(frame[value_column], errors="coerce")
    if signal.notna().mean() < 0.90:
        raise SystemExit(f"numeric signal 비율이 90% 미만: {path}")
    signal.index = frame["feature"].astype(str)
    return signal, value_column


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    qdnaseq_path = args.qdnaseq.resolve()
    run_dir = args.run_dir.resolve()

    config = load_config(config_path)
    actual_depth = str(config.get("depth_label", ""))
    if actual_depth != args.depth:
        raise SystemExit(
            f"depth/config 불일치: --depth={args.depth}, depth_label={actual_depth}"
        )
    if not qdnaseq_path.is_file() and not args.dry_run:
        raise SystemExit(f"QDNAseq 결과 없음: {qdnaseq_path}")

    paths = config["paths"]
    reference_path = Path(
        paths.get(
            "pon_reference",
            Path(paths["work_dir"])
            / "pon_reference"
            / f"pon_reference_{args.depth}.npz",
        )
    ).resolve()
    if not reference_path.is_file():
        raise SystemExit(
            f"PoN reference 없음: {reference_path}\n"
            f"먼저 00_build_pon_reference.py --depth {args.depth} 를 실행하세요."
        )

    matrix_path = run_dir / "matrix" / f"sample_{args.depth}.tsv"
    derived_config_path = run_dir / "config.yaml"
    calling_dir = run_dir / "calling"

    print(f"[stage=qdnaseq] aid={args.aid} depth={args.depth}")
    print(f"qdnaseq:  {qdnaseq_path}")
    print(f"reference:{reference_path}")
    print(f"matrix:   {matrix_path}")

    if args.dry_run:
        print("DRY-RUN: target-only matrix/YAML을 생성하지 않았습니다.")
        return

    validate_meta(config, args.aid)
    if matrix_path.is_file() and derived_config_path.is_file() and not args.force:
        print("SKIP prepare: target-only matrix와 YAML이 이미 존재합니다.")
        return

    with np.load(reference_path, allow_pickle=False) as reference:
        reference_depth = str(reference["depth_label"].item())
        features = reference["features"].astype(str)
    if reference_depth != args.depth:
        raise SystemExit(
            f"reference depth 불일치: expected={args.depth}, actual={reference_depth}"
        )

    signal, value_column = load_signal(qdnaseq_path)
    aligned = signal.reindex(features)
    matched = int(aligned.notna().sum())
    total = len(features)
    coverage = matched / total if total else 0.0
    if coverage < 0.90:
        raise SystemExit(
            f"QDNAseq/reference feature 유효값 비율이 90% 미만: "
            f"{matched}/{total} ({coverage:.2%})"
        )

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_matrix = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
    pd.DataFrame({args.aid: aligned.values}, index=features).rename_axis(
        "feature"
    ).to_csv(temporary_matrix, sep="\t", na_rep="nan", float_format="%.8g")
    os.replace(temporary_matrix, matrix_path)

    derived = config.copy()
    derived["paths"] = dict(paths)
    derived["paths"]["qdnaseq_matrix"] = str(matrix_path)
    derived["paths"]["pon_reference"] = str(reference_path)
    derived["paths"]["out_dir"] = str(calling_dir)
    derived["pipeline_run"] = {
        "aid": args.aid,
        "depth": args.depth,
        "base_config": str(config_path),
        "source_qdnaseq": str(qdnaseq_path),
        "source_value_column": value_column,
        "pon_reference": str(reference_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_yaml_dump(derived, derived_config_path)

    manifest = {
        "aid": args.aid,
        "depth": args.depth,
        "run_dir": str(run_dir),
        "config": str(derived_config_path),
        "qdnaseq": str(qdnaseq_path),
        "matrix": str(matrix_path),
        "pon_reference": str(reference_path),
        "calling_dir": str(calling_dir),
        "feature_valid": {
            "matched": matched,
            "total": total,
            "coverage": coverage,
        },
    }
    atomic_json_dump(manifest, run_dir / "manifest.json")
    print(f"valid feature: {matched}/{total} ({coverage:.2%})")
    print("prepare complete")


if __name__ == "__main__":
    main()
