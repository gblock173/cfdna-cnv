#!/usr/bin/env python3
"""Materialize an entirely synthetic reference and one-row metadata workbook."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from caller.utils.chr_arm_coords import HG38_CHROM_SIZES


def synthetic_signal():
    rng = np.random.default_rng(7421)
    rows = []
    for chrom in range(1, 23):
        size = HG38_CHROM_SIZES[f"chr{chrom}"]
        for start in range(1, size + 1, 100000):
            end = min(start + 99999, size)
            value = rng.normal(0, 0.04 / 0.6744897501960817)
            if chrom == 1 and 20000001 <= start <= 65000000:
                value += 0.45
            rows.append((f"{chrom}:{start}-{end}", chrom, start, end, value))
    return pd.DataFrame(rows, columns=["feature", "chromosome", "start", "end", "example"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--depth", choices=("20G", "100G", "WGS"), default="100G")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("sample.log2.tsv", "metadata.xlsx", "reference.npz", "config.yaml")
    if any((output / name).exists() for name in names):
        parser.error("Synthetic input files already exist; choose a fresh output directory")
    signal = synthetic_signal()
    signal.to_csv(output / "sample.log2.tsv", sep="\t", index=False, float_format="%.8f")
    metadata = pd.DataFrame({
        "AnalysisID": ["example"], "CancerType": ["synthetic"],
        "Stage": ["NA"], "Age": ["NA"], "Sex": ["NA"],
        "Split_AgeSexStage": ["Test"],
    })
    with pd.ExcelWriter(output / "metadata.xlsx", engine="openpyxl") as writer:
        metadata.to_excel(writer, sheet_name="samples", index=False)
        writer.book.properties.creator = "Yujin Jeon"
        writer.book.properties.lastModifiedBy = "Yujin Jeon"
    n = len(signal)
    # Fixed mathematical arrays, not an estimate from individuals or a cohort.
    np.savez_compressed(
        output / "reference.npz",
        format_version=np.asarray(1, dtype=np.int16),
        depth_label=np.asarray(args.depth),
        source_kind=np.asarray("synthetic analytical reference; no human controls"),
        features=signal.feature.to_numpy(dtype=str),
        pon_median=np.zeros(n, dtype=np.float32),
        pon_mad=np.full(n, 0.04, dtype=np.float32),
        raw_q5=np.full(n, -0.097545, dtype=np.float32),
        raw_q95=np.full(n, 0.097545, dtype=np.float32),
        z_q5=np.full(n, -1.0, dtype=np.float32),
        z_q95=np.full(n, 1.0, dtype=np.float32),
        pon_n_total=np.asarray(0), pon_n_effective=np.asarray(0),
        smooth_window=np.asarray(11),
    )
    cfg = yaml.safe_load((ROOT / "configs" / f"config_{args.depth}.example.yaml").read_text())
    cfg["paths"] = {
        "work_dir": str(output), "meta_xlsx": str(output / "metadata.xlsx"),
        "meta_sheet": "samples", "qdnaseq_matrix": str(output / "sample.log2.tsv"),
        "pon_reference": str(output / "reference.npz"),
        "out_dir": str(output / "unused"),
    }
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Synthetic inputs ready: {output}")


if __name__ == "__main__":
    main()
