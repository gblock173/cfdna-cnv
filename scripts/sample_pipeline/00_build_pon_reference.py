#!/usr/bin/env python3
"""Precompute compact depth-specific PoN statistics for single-sample runs."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from caller.utils.config_loader import load_config


DEPTH_CONFIG = {
    "100G": "config.yaml",
    "20G": "config_20G.yaml",
    "WGS": "config_WGS.yaml",
}
CHR_ORDER = [str(i) for i in range(1, 23)]


def median_abs_deviation(values: np.ndarray, axis=None) -> np.ndarray:
    median = np.nanmedian(values, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(values - median), axis=axis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="전체 PoN matrix를 한 번 읽어 재사용 가능한 NPZ 통계를 생성합니다."
    )
    parser.add_argument("--depth", required=True, choices=tuple(DEPTH_CONFIG))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fast_mode(values: np.ndarray, lo: float = -0.3, hi: float = 0.3,
              width: float = 0.005) -> float:
    selected = values[np.isfinite(values) & (values > lo) & (values < hi)]
    if len(selected) < 100:
        return 0.0
    edges = np.arange(lo, hi + width, width)
    histogram, _ = np.histogram(selected, bins=edges)
    smoothed = np.convolve(histogram, np.ones(3) / 3, mode="same")
    peaks = np.where(
        (smoothed[1:-1] > smoothed[:-2])
        & (smoothed[1:-1] >= smoothed[2:])
        & (smoothed[1:-1] >= smoothed.max() * 0.5))[0] + 1
    if len(peaks) == 0:
        return float(((edges[:-1] + edges[1:]) / 2)[np.argmax(smoothed)])
    centers = (edges[:-1] + edges[1:]) / 2
    return float(centers[peaks[np.argmax(smoothed[peaks])]])


def robust_anchor(values: np.ndarray, tolerance: float) -> float:
    mode = fast_mode(values)
    median = float(np.nanmedian(values))
    return mode if abs(mode - median) < tolerance else median


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=window // 2)
        .median()
        .values
    )


def map_aid_to_matrix_column(columns: list[str], meta_aids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        parts = column.split("_")
        first = parts[0]
        if first in meta_aids:
            mapping[first] = column
            continue
        for count in range(2, len(parts) + 1):
            candidate = "_".join(parts[:count])
            if candidate in meta_aids:
                mapping[candidate] = column
                break
    return mapping


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = (args.config or repo / DEPTH_CONFIG[args.depth]).resolve()
    config = load_config(config_path)

    actual_depth = str(config.get("depth_label", ""))
    if actual_depth != args.depth:
        raise SystemExit(
            f"depth/config 불일치: --depth={args.depth}, depth_label={actual_depth}"
        )

    paths = config["paths"]
    matrix_path = Path(paths["qdnaseq_matrix"]).resolve()
    meta_path = Path(paths["meta_xlsx"]).resolve()
    output_path = (
        args.output
        or paths.get("pon_reference")
        or Path(paths["work_dir"]) / "pon_reference" / f"pon_reference_{args.depth}.npz"
    )
    output_path = Path(output_path).resolve()
    split_column = config.get("splits", {}).get(
        "primary_split_col", "Split_AgeSexStage"
    )
    sheet = paths.get("meta_sheet", "전체")
    caller = config.get("caller", {})
    anchor_tolerance = float(caller.get("anchor_tol", 0.02))
    mad_percentile = float(caller.get("pon_mad_percentile_max", 90))
    smooth_window = int(caller.get("smooth_window", 11))

    print(f"[reference] depth={args.depth}")
    print(f"matrix: {matrix_path}")
    print(f"meta:   {meta_path}")
    print(f"output: {output_path}")

    if output_path.is_file() and not args.force:
        print("SKIP: reference가 이미 존재합니다. 다시 만들려면 --force")
        return
    if args.dry_run:
        print("DRY-RUN: reference를 생성하지 않았습니다.")
        return
    if not matrix_path.is_file():
        raise SystemExit(f"PoN 원본 matrix 없음: {matrix_path}")
    if not meta_path.is_file():
        raise SystemExit(f"meta XLSX 없음: {meta_path}")

    meta = pd.read_excel(
        meta_path,
        sheet_name=sheet,
        usecols=["AnalysisID", "CancerType", split_column],
    )
    meta["AnalysisID"] = meta["AnalysisID"].astype(str)
    meta_aids = set(meta["AnalysisID"])
    healthy_train = set(
        meta.loc[
            (meta["CancerType"] == "healthy") & (meta[split_column] == "Train"),
            "AnalysisID",
        ]
    )

    with matrix_path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
    matrix_columns = [column for column in header if column != "feature"]
    aid_to_column = map_aid_to_matrix_column(matrix_columns, meta_aids)
    pon_columns = [aid_to_column[aid] for aid in healthy_train if aid in aid_to_column]
    if not pon_columns:
        raise SystemExit("matrix와 meta가 겹치는 healthy Train PoN이 없습니다.")

    print(f"healthy Train PoN columns: {len(pon_columns)}")
    matrix = pd.read_csv(
        matrix_path,
        sep="\t",
        index_col=0,
        usecols=["feature"] + pon_columns,
    )
    chromosomes = matrix.index.str.extract(r"^(\w+):")[0]
    starts = matrix.index.str.extract(r":(\d+)-")[0].astype(int)
    autosome = chromosomes.isin(CHR_ORDER)
    chromosome_category = pd.Categorical(
        chromosomes[autosome], categories=CHR_ORDER, ordered=True
    )
    sort_index = np.lexsort([starts.values[autosome], chromosome_category.codes])
    features = matrix.index.values[autosome][sort_index].astype(str)
    raw = matrix.values[autosome][sort_index].T.astype(np.float32)
    del matrix

    anchored = np.empty_like(raw)
    raw_mads = np.empty(raw.shape[0], dtype=np.float32)
    for index in range(raw.shape[0]):
        anchor = robust_anchor(raw[index], anchor_tolerance)
        anchored[index] = raw[index] - anchor
        finite = anchored[index][np.isfinite(anchored[index])]
        raw_mads[index] = (
            median_abs_deviation(finite)
            if len(finite)
            else np.inf
        )

    mad_threshold = float(np.percentile(raw_mads, mad_percentile))
    clean = np.isfinite(raw_mads) & (raw_mads <= mad_threshold)
    clean_raw = raw[clean]
    clean_anchored = anchored[clean]
    if clean.sum() < 20:
        raise SystemExit(f"noise filter 후 PoN이 너무 적음: {int(clean.sum())}")

    pon_median = np.nanmedian(clean_anchored, axis=0).astype(np.float32)
    pon_mad = median_abs_deviation(clean_anchored, axis=0
    ).astype(np.float32)
    pon_mad = np.where(pon_mad < 0.005, 0.005, pon_mad).astype(np.float32)
    raw_q5 = np.nanpercentile(clean_raw, 5, axis=0).astype(np.float32)
    raw_q95 = np.nanpercentile(clean_raw, 95, axis=0).astype(np.float32)

    band_count = min(100, clean_anchored.shape[0])
    z_band = np.asarray(
        [
            rolling_median(
                (clean_anchored[index] - pon_median) / pon_mad,
                smooth_window,
            )
            for index in range(band_count)
        ],
        dtype=np.float32,
    )
    z_q5 = np.nanpercentile(z_band, 5, axis=0).astype(np.float32)
    z_q95 = np.nanpercentile(z_band, 95, axis=0).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.asarray(1, dtype=np.int16),
                depth_label=np.asarray(args.depth),
                features=features,
                pon_median=pon_median,
                pon_mad=pon_mad,
                raw_q5=raw_q5,
                raw_q95=raw_q95,
                z_q5=z_q5,
                z_q95=z_q95,
                pon_n_total=np.asarray(raw.shape[0], dtype=np.int32),
                pon_n_effective=np.asarray(clean.sum(), dtype=np.int32),
                pon_mad_percentile=np.asarray(mad_percentile, dtype=np.float32),
                pon_mad_threshold=np.asarray(mad_threshold, dtype=np.float32),
                smooth_window=np.asarray(smooth_window, dtype=np.int16),
                source_matrix=np.asarray(str(matrix_path)),
                source_config=np.asarray(str(config_path)),
                created_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
            )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        f"saved: {output_path}\n"
        f"features={len(features)} PoN={raw.shape[0]} effective={int(clean.sum())} "
        f"MAD_Q{mad_percentile:g}={mad_threshold:.5f}"
    )


if __name__ == "__main__":
    main()
