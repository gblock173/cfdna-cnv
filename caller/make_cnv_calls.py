#!/usr/bin/env python3
"""
make_cnv_calls.py
sample_metrics_<depth>.tsv 를 읽어 calling category + 위치 요약 파일 생성.

출력 컬럼:
  AnalysisID, CancerType, Stage, <split_col>
  label          : CNV+_strong / CNV+_weak / QC_noisy_weak / CNV_suspect / Focal_only / HN / Normal
  cnv_arms       : arm_positive_list  (arm-level CNV 위치, chr.arm+방향, comma-join)
  sig_seg_arms   : significant non-focal segment 위치
  focal_arms     : focal amplification/deletion 위치
  noise_level    : bin_z_MAD
  noise_factor   : sample noise scaling factor
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config_loader import load_config
from labeling import classify, LABEL_ORDER  # noqa: E402

CFG = load_config()
print(f"[config] {CFG['_config_path']}")

WORK_DIR    = Path(CFG["paths"]["work_dir"])
OUT_DIR     = Path(CFG["paths"].get("out_dir",
                                    WORK_DIR / "feature_matrix" / "batch_check" / "cnv_burden"))
DEPTH_LABEL = CFG.get("depth_label", "100G")
SPLIT_COL   = CFG.get("splits", {}).get("primary_split_col", "Split_AgeSexStage")

_CALLER_CFG    = CFG.get("caller", {})
NOISY_BIN_MAD  = float(_CALLER_CFG.get("noisy_bin_mad", 1.81))
NOISE_FACTOR_MAX = float(_CALLER_CFG.get("noise_factor_max", 1.2))
_WEAK_NOISE_CFG = _CALLER_CFG.get("weak_noise_train_oof", {})
WEAK_NOISE_ENABLED = bool(_WEAK_NOISE_CFG.get("enabled", False))
WEAK_NOISE_MAD = (float(_WEAK_NOISE_CFG["noise_mad_threshold"])
                  if WEAK_NOISE_ENABLED else None)
_WEAK_RESCUE_RAW = _WEAK_NOISE_CFG.get("rescue_peak_snr")
WEAK_RESCUE_PEAK_SNR = (float(_WEAK_RESCUE_RAW)
                        if WEAK_NOISE_ENABLED and _WEAK_RESCUE_RAW is not None
                        else None)

IN_TSV  = OUT_DIR / f"sample_metrics_{DEPTH_LABEL}.tsv"
OUT_TSV = OUT_DIR / f"cnv_calls_{DEPTH_LABEL}.tsv"

# ── 로드 & 분류 ──────────────────────────────────────────────────────────────

df = pd.read_csv(IN_TSV, sep="\t")
print(f"loaded: {len(df)} samples from {IN_TSV.name}")

required = ["n_seg_significant", "n_positive_arms", "bin_z_MAD",
            "n_seg_focal", "n_seg_standard", "n_seg_long",
            "n_seg_weak_gain", "n_seg_weak_loss", "noise_factor_capped",
            "n_arms_strong", "n_arms_wide",
            "n_diluted_gain", "n_diluted_loss", "n_diluted_conc_gain",
            "arm_positive_list"]
miss = [c for c in required if c not in df.columns]
if miss:
    raise SystemExit(f"필수 컬럼 누락 — compute_sample_metrics.py 재실행 필요: {miss}")

need_rerun = [c for c in ("sig_seg_arm_list", "focal_seg_arm_list") if c not in df.columns]
if need_rerun:
    raise SystemExit(
        f"컬럼 {need_rerun} 없음.\n"
        f"compute_sample_metrics.py 를 업데이트 버전으로 재실행 후 다시 실행하세요.")

df["label"] = df.apply(
    lambda r: classify(
        r,
        noisy_bin_mad=NOISY_BIN_MAD,
        weak_noise_mad=WEAK_NOISE_MAD,
        weak_rescue_peak_snr=WEAK_RESCUE_PEAK_SNR,
    ),
    axis=1,
)

# ── 출력 컬럼 구성 ────────────────────────────────────────────────────────────

meta_cols = ["AnalysisID", "CancerType", "Stage", SPLIT_COL]
meta_cols = [c for c in meta_cols if c in df.columns]

def clean_list(val):
    if pd.isna(val):
        return ""
    return str(val)

out = df[meta_cols].copy()
out["label"]        = df["label"]
out["cnv_arms"]     = df["arm_positive_list"].apply(clean_list)
out["sig_seg_arms"] = df["sig_seg_arm_list"].apply(clean_list)
out["focal_arms"]   = df["focal_seg_arm_list"].apply(clean_list)
out["noise_level"]  = df["bin_z_MAD"].round(4)
out["noise_factor"] = df["noise_factor"].round(4)

out.to_csv(OUT_TSV, sep="\t", index=False)
print(f"저장: {OUT_TSV}  ({len(out)} rows)")

# ── 요약 출력 ─────────────────────────────────────────────────────────────────

print(f"\n=== {DEPTH_LABEL} label counts ===")
print(df["label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())

if SPLIT_COL in df.columns:
    print(f"\n=== Split x Label ===")
    ct = pd.crosstab(df[SPLIT_COL], df["label"]).reindex(columns=LABEL_ORDER, fill_value=0)
    print(ct.to_string())

# focal 샘플 목록 미리보기
focal_df = out[out["focal_arms"] != ""].sort_values("AnalysisID")
print(f"\n=== Focal amplification 샘플 (n={len(focal_df)}) ===")
if len(focal_df) > 0:
    preview_cols = [c for c in ["AnalysisID","CancerType","Stage","label",
                                 "focal_arms","cnv_arms","noise_level"] if c in focal_df.columns]
    print(focal_df[preview_cols].head(30).to_string(index=False))
