#!/usr/bin/env python3
"""Export the locked CNV rule label and binary prediction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


POSITIVE_LABELS = frozenset({"CNV+_strong", "CNV+_weak"})
VALID_LABELS = frozenset({
    "CNV+_strong", "CNV+_weak", "QC_noisy_weak", "CNV_suspect",
    "Focal_only", "HN", "Normal",
})
OPTIONAL_COLUMNS = (
    "CancerType", "Stage", "Split_AgeSexStage", "Split_ClinicalStrict",
)


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("입력 형식은 DEPTH=/path/sample_labels.tsv")
    depth, raw_path = value.split("=", 1)
    if depth not in {"20G", "100G", "WGS"}:
        raise argparse.ArgumentTypeError(f"지원하지 않는 depth: {depth}")
    if not raw_path:
        raise argparse.ArgumentTypeError("label TSV 경로가 비어 있습니다.")
    return depth, Path(raw_path)


def build_label_pred(inputs: list[tuple[str, Path]], version: str) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    seen_depths: set[str] = set()
    for depth, path in inputs:
        if depth in seen_depths:
            raise ValueError(f"depth 입력 중복: {depth}")
        seen_depths.add(depth)
        if not path.is_file():
            raise FileNotFoundError(f"sample label TSV 없음: {path}")
        frame = pd.read_csv(path, sep="\t")
        missing = {"AnalysisID", "label"}.difference(frame.columns)
        if missing:
            raise ValueError(f"필수 컬럼 누락 {sorted(missing)}: {path}")
        ids = frame["AnalysisID"].astype(str)
        if ids.duplicated().any():
            duplicated = ids[ids.duplicated(keep=False)].unique()
            raise ValueError(
                f"AnalysisID 중복 ({len(duplicated)}개): {', '.join(duplicated[:5])}"
            )
        labels = set(frame["label"].dropna().astype(str))
        unknown = sorted(labels.difference(VALID_LABELS))
        if unknown:
            raise ValueError(f"알 수 없는 label {unknown}: {path}")
        out = pd.DataFrame({
            "version": version,
            "AnalysisID": ids,
            "depth": depth,
            "label": frame["label"].astype(str),
            "pred": frame["label"].isin(POSITIVE_LABELS).astype("int8"),
        })
        for column in OPTIONAL_COLUMNS:
            if column in frame.columns:
                out[column] = frame[column]
        outputs.append(out)
    if not outputs:
        raise ValueError("최소 하나의 --input이 필요합니다.")
    return pd.concat(outputs, ignore_index=True)


def write_atomic_tsv(frame: pd.DataFrame, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        frame.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "확정 CNV label TSV에서 label/pred 결과를 생성합니다. "
            "pred=1은 CNV+_strong 또는 CNV+_weak입니다."
        )
    )
    parser.add_argument("--input", action="append", required=True,
                        type=parse_input, metavar="DEPTH=TSV")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_label_pred(args.input, args.version)
    write_atomic_tsv(output, args.output)
    counts = output.groupby(["depth", "label"], sort=False).size().rename("n")
    print(f"saved: {args.output.resolve()} ({len(output)} rows)")
    print(counts.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
