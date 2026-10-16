#!/usr/bin/env python3
"""Run one sample locally, from QDNAseq log2 bins or a user-supplied BAM."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from caller.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--aid", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qdnaseq", type=Path)
    source.add_argument("--bam", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.aid):
        parser.error("--aid must be a simple sample identifier, without path separators")
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    depth = str(cfg.get("depth_label", ""))
    if depth not in {"20G", "100G", "WGS"}:
        parser.error("config depth_label must be 20G, 100G or WGS")
    for key in ("meta_xlsx", "pon_reference"):
        if not Path(cfg["paths"][key]).is_file():
            parser.error(f"Missing {key}: {cfg['paths'][key]}")
    supplied = (args.bam or args.qdnaseq).resolve()
    if not supplied.is_file():
        parser.error(f"Input file does not exist: {supplied}")
    run_dir = args.out_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error("--out-dir must be empty; use a new directory for each run")
    qdnaseq = supplied
    commands = []
    if args.bam:
        qdnaseq = run_dir / "qdnaseq" / f"{args.aid}.100kb.cnv.log2.txt"
        commands.append([
            args.rscript, str(ROOT / "01_preprocess/extract_CNR.R"),
            "--bam", str(supplied), "--id", args.aid,
            "--binSize", "100", "--outDir", str(qdnaseq.parent),
        ])
    commands.append([
        sys.executable, str(ROOT / "scripts/sample_pipeline/02_prepare_sample_input.py"),
        "--aid", args.aid, "--qdnaseq", str(qdnaseq), "--depth", depth,
        "--config", str(config_path), "--run-dir", str(run_dir),
    ])
    for stage in ("compute_sample_metrics.py", "sample_labeling.py", "make_cnv_calls.py"):
        commands.append([sys.executable, str(ROOT / "caller" / stage)])
    final = run_dir / "label_pred.tsv"
    commands.append([
        sys.executable, str(ROOT / "scripts/final_pipeline/export_label_pred.py"),
        "--input", f"{depth}={run_dir / 'calling' / ('sample_labels_' + depth + '.tsv')}",
        "--version", (ROOT / "VERSION").read_text().strip(), "--output", str(final),
    ])
    if args.profile:
        commands.append([
            sys.executable, str(ROOT / "profiles/profile_one_sample.py"),
            "--aid", args.aid, "--outdir", str(run_dir / "profile"),
        ])
    for cmd in commands:
        print("COMMAND:", repr(cmd), flush=True)
    if args.plan_only:
        return
    env = os.environ.copy()
    env["CFDNA_CNV_CONFIG"] = str(run_dir / "config.yaml")
    for cmd in commands:
        subprocess.run(cmd, env=env, check=True)
    print(f"FINAL_OUTPUT: {final}")


if __name__ == "__main__":
    main()
