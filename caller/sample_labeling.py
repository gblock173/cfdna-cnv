#!/usr/bin/env python3
"""
sample_labeling.py
7-label classification (segment + arm hybrid + suspect + Train-OOF noise gate):
  seg_strong  = (n_seg_standard >= 1) OR (n_seg_long >= 1)   [baseline nf]
  arm_strong  = (n_arms_strong  >= 1) OR (n_arms_wide >= 1)
  seg_weak    = (n_seg_weak_gain >= 1) OR (n_seg_weak_loss >= 1)  [nf cap tier]
  noisy       = (bin_z_MAD >= NOISY_BIN_MAD)

  if seg_strong AND arm_strong:                   -> "CNV+_strong"
  elif seg_weak AND Train-OOF noise gate fails:   -> "QC_noisy_weak"
  elif seg_weak:                                  -> "CNV+_weak"
  elif seg_strong OR  arm_strong:                 -> "CNV_suspect"
  elif n_seg_focal >= 1:                          -> "Focal_only"
  elif noisy:                                     -> "HN"
  else:                                           -> "Normal"

출력:
  - sample_labels_<depth>.tsv (all metrics + label + noisy_flag)
  - sample_labels_scatter.png  (7-color scatter + threshold lines)
  - sample_labels_by_cancer.png (stacked bar by CancerType)
  - sample_labels_by_stage.png  (stacked bar by Stage, cancer only)
"""

import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config_loader import load_config  # noqa: E402

CFG = load_config()
print(f"[config] {CFG['_config_path']}")

WORK_DIR    = Path(CFG["paths"]["work_dir"])
OUT_DIR     = Path(CFG["paths"].get("out_dir",
                                    WORK_DIR / "feature_matrix" / "batch_check" / "cnv_burden"))
DEPTH_LABEL = CFG.get("depth_label", "100G")
TSV         = OUT_DIR / f"sample_metrics_{DEPTH_LABEL}.tsv"
SPLIT_COL   = CFG.get("splits", {}).get("primary_split_col", "Split_AgeSexStage")

# hybrid 분류 임계값 (segment/arm tier 정의는 compute_sample_metrics.py와 일치)
SEG_TIERS = [
    {"name": "focal",    "L_min": 5,   "L_max": 30,  "amp_min": 7.0, "peak_min": 10.0,
     "density_min": 0.50, "snr_min": 5.0, "peak_snr_min": 7.0},
    {"name": "standard", "L_min": 30,  "amp_min": 4.5, "peak_min": 5.0,  "density_min": 0.30},
    {"name": "long",     "L_min": 300, "amp_min": 2.0, "peak_min": 3.0,  "density_min": 0.10},
]
ARM_TIERS = [
    {"name": "strong", "mean_min": 2.5, "frac_min": 0.30},
    {"name": "wide",   "mean_min": 1.2, "frac_min": 0.50},
]

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

# scatter overlay 용 대표 임계값 (가장 보수적인 tier)
SEG_PEAK_REF  = min(t["peak_min"] for t in SEG_TIERS)   # = 3.0
ARM_MEAN_REF  = min(t["mean_min"] for t in ARM_TIERS)   # = 1.2

from labeling import classify, is_seg_pos, is_arm_pos, is_seg_weak, has_diluted_raw  # noqa: E402
from labeling import LABEL_ORDER, LABEL_COLOR  # noqa: E402


df = pd.read_csv(TSV, sep="\t")
print(f"loaded: {len(df)}")

required = ["n_seg_significant","n_positive_arms","bin_z_MAD",
            "max_seg_peak_abs_z","max_arm_abs_mean_z","max_arm_frac_outlier",
            "n_seg_focal","n_seg_standard","n_seg_long",
            "n_seg_weak_gain","n_seg_weak_loss","noise_factor_capped",
            "n_arms_strong","n_arms_wide","noise_factor",
            "n_diluted_gain","n_diluted_loss",
            "n_diluted_conc_gain","n_diluted_conc_loss"]
miss = [c for c in required if c not in df.columns]
if miss:
    raise SystemExit(f"필수 컬럼 누락 (compute_sample_metrics.py 재실행 필요): {miss}")

df["label"] = df.apply(
    lambda r: classify(
        r,
        noisy_bin_mad=NOISY_BIN_MAD,
        weak_noise_mad=WEAK_NOISE_MAD,
        weak_rescue_peak_snr=WEAK_RESCUE_PEAK_SNR,
    ),
    axis=1,
)
df["noisy_flag"] = (df["bin_z_MAD"] >= NOISY_BIN_MAD).astype(int)
df["weak_noise_train_oof_flag"] = (df["label"] == "QC_noisy_weak").astype(int)
df["group"]      = "Cancer"
df.loc[(df["CancerType"]=="healthy") & (df[SPLIT_COL]=="Train"), "group"] = "Healthy (Train, PoN)"
df.loc[(df["CancerType"]=="healthy") & (df[SPLIT_COL]!="Train"), "group"] = "Healthy (Test/Val)"

out_tsv = OUT_DIR / f"sample_labels_{DEPTH_LABEL}.tsv"
df.to_csv(out_tsv, sep="\t", index=False)
print(f"saved: {out_tsv}")

print("\n=== Overall label counts ===")
print(df["label"].value_counts())

print("\n=== Group x Label ===")
ct = pd.crosstab(df["group"], df["label"]).reindex(columns=LABEL_ORDER, fill_value=0)
print(ct.to_string())
print("\nPercentages:")
print(ct.div(ct.sum(axis=1), axis=0).mul(100).round(1).to_string())

# Healthy Train 합격기준 검증 (CNV+_strong/weak 누수율, CNV_suspect 별도)
htrain = df[df["group"]=="Healthy (Train, PoN)"]
if len(htrain) > 0:
    vc = htrain["label"].value_counts()
    n_strong  = int(vc.get("CNV+_strong", 0))
    n_weak    = int(vc.get("CNV+_weak",   0))
    n_suspect = int(vc.get("CNV_suspect", 0))
    n_hn      = int(vc.get("HN",          0))
    n_pos     = n_strong + n_weak
    total     = len(htrain)
    print(f"\n[합격기준] Healthy Train CNV+ (strong+weak): {n_pos}/{total} = {n_pos/total*100:.2f}%   (목표 <= 1%)")
    print(f"            CNV_suspect: {n_suspect}/{total} = {n_suspect/total*100:.2f}%   (참고)")
    print(f"            HN:          {n_hn}/{total} = {n_hn/total*100:.2f}%   (참고)")

# HN→CNV+_weak 누수 가드: noisy 샘플이 CNV+_weak 로 잘못 분류되는지 모니터링
noisy_df  = df[df["bin_z_MAD"] >= NOISY_BIN_MAD]
if len(noisy_df) > 0:
    n_noisy_w  = int((noisy_df["label"]=="CNV+_weak").sum())
    n_noisy_su = int((noisy_df["label"]=="CNV_suspect").sum())
    print(f"\n[가드] noisy(bin_z_MAD>={NOISY_BIN_MAD}) n={len(noisy_df)} 중  "
          f"CNV+_weak={n_noisy_w}({n_noisy_w/len(noisy_df)*100:.1f}%)  "
          f"CNV_suspect={n_noisy_su}({n_noisy_su/len(noisy_df)*100:.1f}%)")

print("\n=== CancerType x Label ===")
ctc = pd.crosstab(df["CancerType"], df["label"]).reindex(columns=LABEL_ORDER, fill_value=0)
print(ctc.to_string())

print("\n=== Stage x Label (cancer only) ===")
cancer_only = df[df["CancerType"]!="healthy"].copy()
cts = pd.crosstab(cancer_only["Stage"].fillna("NA"), cancer_only["label"]).reindex(columns=LABEL_ORDER, fill_value=0)
print(cts.to_string())

# ─────────── 시각화 1: 6-color scatter (arm mean vs seg peak) ───────────
fig = plt.figure(figsize=(18, 8))
gs  = fig.add_gridspec(1, 2, wspace=0.25)

ax1 = fig.add_subplot(gs[0,0])
for lbl in LABEL_ORDER:
    sub = df[df["label"]==lbl]
    ax1.scatter(sub["max_arm_abs_mean_z"], sub["max_seg_peak_abs_z"],
                c=LABEL_COLOR[lbl], s=8, alpha=0.55, linewidths=0,
                label=f"{lbl} (n={len(sub)})")
for t in ARM_TIERS:
    ax1.axvline(t["mean_min"], color="black", lw=0.6, ls="--", alpha=0.5)
    ax1.text(t["mean_min"]+0.05, ax1.get_ylim()[1]*0.02,
             f"arm_{t['name']}={t['mean_min']}", fontsize=7)
for t in SEG_TIERS:
    ax1.axhline(t["peak_min"], color="black", lw=0.6, ls="--", alpha=0.5)
    ax1.text(0.02, t["peak_min"]+0.15, f"seg_{t['name']}_peak={t['peak_min']}", fontsize=7)
ax1.set_xlabel("max_arm_abs_mean_z", fontsize=10)
ax1.set_ylabel("max_seg_peak_abs_z", fontsize=10)
ax1.set_title("(a) strong=seg+arm, weak=nf-capped seg, suspect=one-side only",
              fontsize=11, fontweight="bold")
ax1.set_xlim(0, max(df["max_arm_abs_mean_z"].quantile(0.999)*1.05, ARM_MEAN_REF*3))
ax1.set_ylim(0, max(df["max_seg_peak_abs_z"].quantile(0.999)*1.05, SEG_PEAK_REF*3))
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0,1])
for lbl in LABEL_ORDER:
    sub = df[df["label"]==lbl]
    ax2.scatter(sub["n_positive_arms"], sub["n_seg_significant"],
                c=LABEL_COLOR[lbl], s=8, alpha=0.55, linewidths=0,
                label=f"{lbl} (n={len(sub)})")
ax2.axvline(0.5, color="black", lw=0.6, ls="--", alpha=0.6)
ax2.axhline(0.5, color="black", lw=0.6, ls="--", alpha=0.6)
ax2.set_xlabel("n_positive_arms (arm-level shift count)", fontsize=10)
ax2.set_ylabel("n_seg_significant (focal segment count)", fontsize=10)
ax2.set_title("(b) n_positive_arms vs n_seg_significant (count plane)",
              fontsize=11, fontweight="bold")
ax2.set_xlim(-0.5, max(df["n_positive_arms"].quantile(0.999)+1, 5))
ax2.set_ylim(-0.5, max(df["n_seg_significant"].quantile(0.999)+1, 5))
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(alpha=0.3)

seg_label = " | ".join(f"{t['name']}(L>={t['L_min']},amp>={t['amp_min']},peak>={t['peak_min']},den>={t['density_min']})"
                       for t in SEG_TIERS)
arm_label = " | ".join(f"{t['name']}(|mean|>={t['mean_min']},frac>={t['frac_min']})"
                       for t in ARM_TIERS)
fig.suptitle(f"Hybrid 분류 — weak seg: nf cap={NOISE_FACTOR_MAX} std/long    "
             f"seg tiers: {seg_label}    arm tiers: {arm_label}",
             fontsize=11, fontweight="bold", y=1.00)
plt.tight_layout()
out1 = OUT_DIR / "sample_labels_scatter.png"
plt.savefig(out1, dpi=130, bbox_inches="tight")
INTRO_DIR = OUT_DIR / "cnv_caller_intro"; INTRO_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(INTRO_DIR / "perf_labels_scatter.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"\n저장: {out1}")

# ─────────── 시각화 2: CancerType별 stacked bar ───────────
ctc_pct = ctc.div(ctc.sum(axis=1), axis=0).mul(100)
order = ctc.sum(axis=1).sort_values(ascending=False).index.tolist()
ctc_pct = ctc_pct.loc[order]

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
ctc.loc[order].plot(kind="bar", stacked=True, ax=axes[0],
                     color=[LABEL_COLOR[l] for l in LABEL_ORDER], width=0.75)
axes[0].set_title("CancerType별 라벨 분포 (count)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("# samples"); axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=45, labelsize=9)
axes[0].legend(fontsize=9, loc="upper right")

ctc_pct.plot(kind="bar", stacked=True, ax=axes[1],
              color=[LABEL_COLOR[l] for l in LABEL_ORDER], width=0.75)
axes[1].set_title("CancerType별 라벨 분포 (%)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("%"); axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=45, labelsize=9)
axes[1].set_ylim(0,100)
axes[1].legend(fontsize=9, loc="upper right")

fig.suptitle("Hybrid 분류 by CancerType (PoN=healthy Train)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
out2 = OUT_DIR / "sample_labels_by_cancer.png"
plt.savefig(out2, dpi=130, bbox_inches="tight")
plt.savefig(INTRO_DIR / "perf_labels_by_cancer.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"저장: {out2}")

# ─────────── 시각화 3: Stage별 stacked bar (cancer only) ───────────
cts_pct = cts.div(cts.sum(axis=1), axis=0).mul(100)
stage_order = sorted([s for s in cts.index if s != "NA"]) + (["NA"] if "NA" in cts.index else [])
cts      = cts.loc[stage_order]
cts_pct  = cts_pct.loc[stage_order]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cts.plot(kind="bar", stacked=True, ax=axes[0],
         color=[LABEL_COLOR[l] for l in LABEL_ORDER], width=0.7)
axes[0].set_title("Stage별 라벨 분포 (count, cancer only)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("# samples"); axes[0].set_xlabel("Stage")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(fontsize=9, loc="upper right")

cts_pct.plot(kind="bar", stacked=True, ax=axes[1],
             color=[LABEL_COLOR[l] for l in LABEL_ORDER], width=0.7)
axes[1].set_title("Stage별 라벨 분포 (%, cancer only)", fontsize=11, fontweight="bold")
axes[1].set_ylabel("%"); axes[1].set_xlabel("Stage")
axes[1].tick_params(axis="x", rotation=0)
axes[1].set_ylim(0,100)
axes[1].legend(fontsize=9, loc="upper right")

fig.suptitle("Hybrid 분류 by Stage (cancer only)", fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()
out3 = OUT_DIR / "sample_labels_by_stage.png"
plt.savefig(out3, dpi=130, bbox_inches="tight")
plt.savefig(INTRO_DIR / "perf_labels_by_stage.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"저장: {out3}")
