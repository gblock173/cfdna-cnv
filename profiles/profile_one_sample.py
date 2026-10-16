#!/usr/bin/env python3
"""Draw raw and standardized single-sample CNV profiles.

Labels and summary metrics are read from caller output. Shaded segment
illustrations are not an exported segmentation or a separate calling result.
Use CFDNA_CNV_CONFIG to select the run configuration and --aid example
for the bundled synthetic example.
"""

import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caller.utils.config_loader import load_config  # noqa: E402
from caller.labeling import classify as caller_classify  # noqa: E402
from caller.segment_utils import (  # noqa: E402
    draw_seg_band,
    legend_elements as seg_legend_elements,
    seg_plot_tier,
    augment_cfdna_diluted_segments,
    NOISE_FACTOR_MAX_DEFAULT,
)

CFG         = load_config()
DEPTH_LABEL = CFG.get("depth_label", "100G")
WORK_DIR    = Path(CFG["paths"]["work_dir"])
QDNASEQ_MAT = Path(CFG["paths"]["qdnaseq_matrix"])
PON_REFERENCE = (Path(CFG["paths"]["pon_reference"])
                 if CFG["paths"].get("pon_reference") else None)
META_XLSX   = Path(CFG["paths"]["meta_xlsx"])
META_SHEET  = CFG["paths"].get("meta_sheet", "전체")
OUT_DIR     = Path(CFG["paths"].get("out_dir",
                                    WORK_DIR / "feature_matrix" / "batch_check" / "cnv_burden"))
NOISE_FACTOR_MAX = float(CFG.get("caller", {}).get("noise_factor_max", NOISE_FACTOR_MAX_DEFAULT))
NOISY_BIN_MAD = float(CFG.get("caller", {}).get("noisy_bin_mad", 1.81))
SPLIT_COL   = CFG.get("splits", {}).get("primary_split_col", "Split_AgeSexStage")
CHR_ORDER   = [str(i) for i in range(1, 23)]
ANCHOR_TOL  = float(CFG.get("caller", {}).get("anchor_tol", 0.02))
WINDOW      = int(CFG.get("caller", {}).get("smooth_window", 11))

_CALLER_CFG = CFG.get("caller", {})
Z_CALL_THR  = float(_CALLER_CFG.get("z_call_thr", 3.0))
Z_LO_THR    = float(_CALLER_CFG.get("z_lo_thr",   1.5))
SEG_GAP_MAX = int(  _CALLER_CFG.get("seg_gap_max", 20))
SEG_TIERS = [
    {"name": "focal",    "L_min": 5,   "L_max": 30,  "amp_min": 7.0, "peak_min": 10.0,
     "density_min": 0.50, "snr_min": 5.0, "peak_snr_min": 7.0},
    {"name": "standard", "L_min": 30,  "amp_min": 4.5, "peak_min": 5.0,  "density_min": 0.30},
    {"name": "long",     "L_min": 300, "amp_min": 2.0, "peak_min": 3.0,  "density_min": 0.10},
]
NOISE_REF_MAD    = 1.0
BG_MIN_BIN_COUNT = 2000
FOCAL_SUBSEG_HI_ABS = 2.0
LABEL_TITLE_COLOR = {
    "CNV+_strong": "#1a9850",
    "CNV+_weak":   "#fdae61",
    "CNV_suspect": "#b8860b",
    "Focal_only":  "#9970ab",
    "HN":          "#d73027",
    "Normal":      "#444",
}


# ── segment/anchor/z 헬퍼 (compute_sample_metrics.py / profile_cancer_cohort.py 와 동일) ──
def sample_noise_factor(bin_z_mad, ref=NOISE_REF_MAD):
    if not np.isfinite(bin_z_mad) or bin_z_mad <= ref:
        return 1.0
    return float(bin_z_mad / ref)


def _seg_passes_tier(s, t, noise_factor=1.0, bg_mad=None):
    if s["length"] < t["L_min"]:
        return False
    if "L_max" in t and s["length"] > t["L_max"]:
        return False
    nf = 1.0 if t["name"] == "focal" else noise_factor
    if not (s["mean_abs_z"] >= t["amp_min"]  * nf
            and s["peak_abs_z"] >= t["peak_min"] * nf
            and s["density"]    >= t["density_min"]):
        return False
    if bg_mad is not None and bg_mad > 0:
        snr_min = t.get("snr_min")
        if snr_min is not None and s["mean_abs_z"] / bg_mad < snr_min:
            return False
        peak_snr_min = t.get("peak_snr_min")
        if peak_snr_min is not None and s["peak_abs_z"] / bg_mad < peak_snr_min:
            return False
    return True


def significant_seg_mask(segments, n_bins, tiers, noise_factor=1.0, bg_mad=None):
    mask = np.zeros(n_bins, dtype=bool)
    for s in segments:
        for t in tiers:
            if _seg_passes_tier(s, t, noise_factor=noise_factor, bg_mad=bg_mad):
                mask[s["start"]:s["end"]+1] = True
                break
    return mask


def background_bin_mad(z_bin, sig_mask, fallback_mad, min_bin_count=BG_MIN_BIN_COUNT):
    bg = z_bin[~sig_mask]
    bg = bg[np.isfinite(bg)]
    if len(bg) < min_bin_count:
        return float(fallback_mad)
    return float(stats.median_abs_deviation(bg, nan_policy="omit"))


def extract_focal_subsegments(segments, z_smooth, bin_chrom, focal_tier, bg_mad,
                              hi_abs=FOCAL_SUBSEG_HI_ABS):
    L_min = focal_tier["L_min"]; L_max = focal_tier["L_max"]
    out = []
    for s in segments:
        if s["length"] <= L_max:
            if _seg_passes_tier(s, focal_tier, noise_factor=1.0, bg_mad=bg_mad):
                continue
        sub = z_smooth[s["start"]:s["end"]+1]
        if not np.isfinite(sub).any():
            continue
        chrom_sub = bin_chrom[s["start"]:s["end"]+1] if bin_chrom is not None else None
        abs_sub = np.abs(sub)
        peak_local = int(np.nanargmax(abs_sub))
        peak_val   = float(abs_sub[peak_local])
        if peak_val < focal_tier["peak_min"]:
            continue
        direction = 1 if sub[peak_local] > 0 else -1
        peak_chrom = chrom_sub[peak_local] if chrom_sub is not None else None
        l = peak_local
        while l > 0:
            c = sub[l-1]
            if not np.isfinite(c): break
            if chrom_sub is not None and chrom_sub[l-1] != peak_chrom: break
            if direction * c < hi_abs: break
            if (peak_local - (l-1) + 1) > L_max: break
            l -= 1
        r = peak_local
        while r < len(sub) - 1:
            c = sub[r+1]
            if not np.isfinite(c): break
            if chrom_sub is not None and chrom_sub[r+1] != peak_chrom: break
            if direction * c < hi_abs: break
            if ((r+1) - l + 1) > L_max: break
            r += 1
        win = sub[l:r+1]
        win_f = win[np.isfinite(win)]
        win_len = r - l + 1
        if win_len < L_min:
            continue
        abs_w = np.abs(win_f)
        mean_a = float(abs_w.mean()); peak_a = float(abs_w.max())
        n_strict = int((direction * win_f >= hi_abs).sum())
        density  = n_strict / win_len if win_len > 0 else 0.0
        cand = {"start": int(s["start"]+l), "end": int(s["start"]+r),
                "direction": int(direction), "length": int(win_len),
                "mean_abs_z": mean_a, "peak_abs_z": peak_a,
                "n_strict": n_strict, "density": float(density),
                "from_subseg": True}
        if _seg_passes_tier(cand, focal_tier, noise_factor=1.0, bg_mad=bg_mad):
            out.append(cand)
    return out


def adaptive_noise_factor_and_segs(z_bin, z_smooth, segments_base, bin_chrom,
                                   tiers=None, fallback_mad=None):
    if tiers is None:
        tiers = SEG_TIERS
    focal_tier = tiers[0]
    n_bins = len(z_bin)
    fb     = fallback_mad if fallback_mad is not None else \
             float(stats.median_abs_deviation(z_bin[np.isfinite(z_bin)], nan_policy="omit"))
    sub_p0 = extract_focal_subsegments(segments_base, z_smooth, bin_chrom, focal_tier, bg_mad=fb)
    sig_p0 = significant_seg_mask(segments_base+sub_p0, n_bins, tiers, noise_factor=1.0, bg_mad=fb)
    bg1    = background_bin_mad(z_bin, sig_p0, fallback_mad=fb)
    nf1    = sample_noise_factor(bg1)
    sub_p1 = extract_focal_subsegments(segments_base, z_smooth, bin_chrom, focal_tier, bg_mad=bg1)
    sig_p1 = significant_seg_mask(segments_base+sub_p1, n_bins, tiers, noise_factor=nf1, bg_mad=bg1)
    bg2    = background_bin_mad(z_bin, sig_p1, fallback_mad=bg1)
    nf2    = sample_noise_factor(bg2)
    sub_p2 = extract_focal_subsegments(segments_base, z_smooth, bin_chrom, focal_tier, bg_mad=bg2)
    return nf2, bg2, segments_base + sub_p2


def fast_mode(v, lo=-0.3, hi=0.3, bw=0.005):
    v2 = v[np.isfinite(v) & (v > lo) & (v < hi)]
    if len(v2) < 100: return 0.0
    edges = np.arange(lo, hi+bw, bw)
    h, _  = np.histogram(v2, bins=edges)
    centers = (edges[:-1]+edges[1:])/2
    h_s = np.convolve(h, np.ones(3)/3, mode="same")
    peaks, _ = find_peaks(h_s, height=h_s.max()*0.5)
    if len(peaks) == 0: return float(centers[np.argmax(h_s)])
    return float(centers[peaks[np.argmax(h_s[peaks])]])


def robust_anchor(v):
    mode_v   = fast_mode(v)
    median_v = float(np.nanmedian(v))
    if abs(mode_v - median_v) < ANCHOR_TOL:
        return mode_v, "mode"
    return median_v, "median"


def rolling_median(v, w=WINDOW):
    return pd.Series(v).rolling(window=w, center=True, min_periods=w//2).median().values


def extract_segments(z_smooth, bin_chrom=None,
                     hi_thr=Z_CALL_THR, lo_thr=Z_LO_THR, gap_max=SEG_GAP_MAX):
    """compute_sample_metrics.py 와 동일 hysteresis 로직."""
    n = len(z_smooth)
    finite = np.isfinite(z_smooth)
    segs = []
    state = 0; seg_start = -1; last_extend = -1; gap_counter = 0; prev_chrom = None

    def _close(direction):
        if seg_start < 0 or last_extend < 0 or last_extend < seg_start:
            return
        sub = z_smooth[seg_start:last_extend+1]
        sub_f = sub[np.isfinite(sub)]
        if len(sub_f) == 0: return
        abs_v = np.abs(sub_f)
        if direction > 0:
            n_strict = int((sub_f >= hi_thr).sum())
        else:
            n_strict = int((sub_f <= -hi_thr).sum())
        length = int(last_extend - seg_start + 1)
        segs.append({"start": int(seg_start), "end": int(last_extend),
                     "direction": int(direction), "length": length,
                     "mean_abs_z": float(abs_v.mean()),
                     "peak_abs_z": float(abs_v.max()),
                     "n_strict":   n_strict,
                     "density":    float(n_strict / length) if length > 0 else 0.0})

    for i in range(n):
        cur_chrom = bin_chrom[i] if bin_chrom is not None else None
        if state != 0 and prev_chrom is not None and cur_chrom is not None and cur_chrom != prev_chrom:
            _close(state); state = 0; seg_start = last_extend = -1; gap_counter = 0
        if not finite[i]:
            if state != 0:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state); state = 0; seg_start = last_extend = -1; gap_counter = 0
            prev_chrom = cur_chrom
            continue
        zv = z_smooth[i]
        if state == 0:
            if zv >= hi_thr:
                state = 1; seg_start = last_extend = i; gap_counter = 0
            elif zv <= -hi_thr:
                state = -1; seg_start = last_extend = i; gap_counter = 0
        elif state == 1:
            if zv <= -hi_thr:
                _close(state); state = -1; seg_start = last_extend = i; gap_counter = 0
            elif zv >= lo_thr:
                last_extend = i; gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state); state = 0; seg_start = last_extend = -1; gap_counter = 0
        else:
            if zv >= hi_thr:
                _close(state); state = 1; seg_start = last_extend = i; gap_counter = 0
            elif zv <= -lo_thr:
                last_extend = i; gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state); state = 0; seg_start = last_extend = -1; gap_counter = 0
        prev_chrom = cur_chrom
    if state != 0:
        _close(state)
    return segs


def parse_args():
    p = argparse.ArgumentParser(description="단일 샘플 CNV z-score profile (label TSV 값과 일치)")
    p.add_argument("--aid", required=True,
                   help="AnalysisID. 쉼표로 여러 개 지정 가능 (예: example)")
    p.add_argument("--ylim-raw", type=float, default=1.0)
    p.add_argument("--ylim-z",   type=float, default=50.0)
    p.add_argument("--suffix", default="oneoff", help="출력 파일명 suffix")
    p.add_argument("--outdir", default=None,
                   help="출력 폴더 (기본: <out_dir>/oneoff_profiles)")
    return p.parse_args()


def main():
    args = parse_args()
    aids = [a.strip() for a in args.aid.split(",") if a.strip()]

    label_tsv = OUT_DIR / f"sample_labels_{DEPTH_LABEL}.tsv"
    if not label_tsv.exists():
        raise SystemExit(f"label TSV 없음: {label_tsv}\n먼저 caller 를 실행해 라벨을 만들어야 합니다.")
    labels = pd.read_csv(label_tsv, sep="\t")

    meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)[
        ["AnalysisID", "CancerType", "Age", "Sex", "Stage", SPLIT_COL]]

    # 대상 샘플 row (공식 caller 출력)
    rows = []
    for aid in aids:
        sub = labels[labels["AnalysisID"] == aid]
        if len(sub) == 0:
            raise SystemExit(f"'{aid}' 가 {label_tsv.name} 에 없습니다. (depth={DEPTH_LABEL})")
        info = sub.iloc[0].to_dict()
        mrow = meta[meta["AnalysisID"] == aid]
        info["Age"] = float(mrow["Age"].iloc[0]) if len(mrow) and pd.notna(mrow["Age"].iloc[0]) else np.nan
        info["Sex"] = mrow["Sex"].iloc[0] if len(mrow) else np.nan
        rows.append(info)

    # ── label 일치성 런타임 검증: classify(row) == row['label'] ──
    print(f"[depth={DEPTH_LABEL}] label 일치성 검증 (classify vs TSV.label):")
    for info in rows:
        gate = _CALLER_CFG.get("weak_noise_train_oof", {})
        enabled = gate.get("enabled", False)
        recomputed = caller_classify(
            info, noisy_bin_mad=NOISY_BIN_MAD,
            weak_noise_mad=gate.get("noise_mad_threshold") if enabled else None,
            weak_rescue_peak_snr=gate.get("rescue_peak_snr") if enabled else None,
        )
        ok = (recomputed == info["label"])
        mark = "OK" if ok else "MISMATCH"
        print(f"  {info['AnalysisID']:10s}  TSV.label={info['label']:11s}  "
              f"classify()={recomputed:11s}  [{mark}]")
        if not ok:
            print(f"    ! 경고: TSV 와 classify 결과 불일치 — config/노이즈 임계값 확인 필요")

    # ── QDNAseq matrix 헤더 → aid→column 매핑 ──
    with open(QDNASEQ_MAT) as f:
        header = f.readline().strip().split("\t")
    meta_aids = set(meta["AnalysisID"].dropna().astype(str))
    aid_to_bam = {}
    for column in [h for h in header if h != "feature"]:
        parts = column.split("_")
        first = parts[0]
        if first in meta_aids:
            aid_to_bam[first] = column
            continue
        for count in range(2, len(parts) + 1):
            candidate = "_".join(parts[:count])
            if candidate in meta_aids:
                aid_to_bam[candidate] = column
                break

    missing = [a for a in aids if a not in aid_to_bam]
    if missing:
        raise SystemExit(f"QDNAseq matrix 에 없는 샘플: {missing}")

    target_bam_ids = [aid_to_bam[a] for a in aids]
    pon_bam_ids = []
    if PON_REFERENCE is None:
        target_set = set(aids)
        train_healthy_aids = set(
            meta[(meta["CancerType"] == "healthy") &
                 (meta[SPLIT_COL] == "Train")]["AnalysisID"])
        pon_bam_ids = [aid_to_bam[a] for a in train_healthy_aids
                       if a in aid_to_bam and a not in target_set]
        print(f"PoN = healthy Train ∩ {DEPTH_LABEL} − target = {len(pon_bam_ids)}")

    usecols = ["feature"] + list(set(target_bam_ids + pon_bam_ids))
    print(f"matrix 로드 ({len(usecols)-1} samples)...")
    mat = pd.read_csv(QDNASEQ_MAT, sep="\t", index_col=0, usecols=usecols)
    mat = mat[~mat.index.str.startswith("X:")]

    chrs = mat.index.str.extract(r"^(\w+):")[0]
    starts = mat.index.str.extract(r":(\d+)-")[0].astype(int)
    mask = chrs.isin(CHR_ORDER)
    chr_cat = pd.Categorical(chrs[mask], categories=CHR_ORDER, ordered=True)
    sort_idx = np.lexsort([starts.values[mask], chr_cat.codes])
    chrs_sorted = chrs[mask].values[sort_idx]
    features_sorted = mat.index.values[mask][sort_idx].astype(str)
    boundaries = {
        c: (np.where(chrs_sorted == c)[0][0],
            np.where(chrs_sorted == c)[0][-1])
        for c in CHR_ORDER if (chrs_sorted == c).any()
    }

    def get_vals(bid):
        # compute_sample_metrics.py 와 동일한 float32 - anchor/PoN 일치 보장
        return mat[bid].values[mask][sort_idx].astype(np.float32)

    if PON_REFERENCE is not None:
        print(f"compact PoN reference 로드: {PON_REFERENCE}")
        if not PON_REFERENCE.is_file():
            raise SystemExit(f"PoN reference 없음: {PON_REFERENCE}")
        with np.load(PON_REFERENCE, allow_pickle=False) as ref:
            ref_depth = str(ref["depth_label"].item())
            ref_features = ref["features"].astype(str)
            pon_median = ref["pon_median"].astype(np.float32)
            pon_mad = ref["pon_mad"].astype(np.float32)
            raw_q5 = ref["raw_q5"].astype(np.float32)
            raw_q95 = ref["raw_q95"].astype(np.float32)
            z_q5 = ref["z_q5"].astype(np.float32)
            z_q95 = ref["z_q95"].astype(np.float32)
            pon_display_n = int(ref["pon_n_effective"].item())
        if ref_depth != DEPTH_LABEL:
            raise SystemExit(
                f"PoN reference depth 불일치: config={DEPTH_LABEL}, "
                f"reference={ref_depth}")
        if not np.array_equal(features_sorted, ref_features):
            raise SystemExit(
                "QDNAseq matrix feature/order가 PoN reference와 일치하지 않습니다.")
    else:
        print("PoN 계산...")
        pon_raw_stack = np.array(
            [get_vals(b) for b in pon_bam_ids if b in mat.columns])
        pon_anc_stack = np.empty_like(pon_raw_stack)
        for i in range(pon_raw_stack.shape[0]):
            anchor, _ = robust_anchor(pon_raw_stack[i])
            pon_anc_stack[i] = pon_raw_stack[i] - anchor

        raw_q5 = np.nanpercentile(pon_raw_stack, 5, axis=0)
        raw_q95 = np.nanpercentile(pon_raw_stack, 95, axis=0)
        pon_median = np.nanmedian(pon_anc_stack, axis=0)
        pon_mad = stats.median_abs_deviation(
            pon_anc_stack, axis=0, nan_policy="omit")
        pon_mad = np.where(pon_mad < 0.005, 0.005, pon_mad)
        z_band = np.array(
            [rolling_median((pon_anc_stack[i] - pon_median) / pon_mad)
             for i in range(min(100, pon_anc_stack.shape[0]))])
        z_q5 = np.nanpercentile(z_band, 5, axis=0)
        z_q95 = np.nanpercentile(z_band, 95, axis=0)
        pon_display_n = pon_raw_stack.shape[0]

    n_bins = len(features_sorted)

    # ── plot ──
    n_plots = len(rows)
    x = np.arange(n_bins)
    ylim_raw = (-args.ylim_raw, args.ylim_raw)
    ylim_z   = (-args.ylim_z,   args.ylim_z)

    # 범례 항목: arrow 계열(seg arrow: ...) 제외 + sample anchor 추가.
    # 미리 만든 뒤 실제 렌더링 높이로 푸터를 산정한다.
    LEG_NCOL = 3
    legend_elements = [e for e in seg_legend_elements(Z_CALL_THR, show_focal=True)
                       if "arrow" not in e.get_label().lower()]
    legend_elements.append(Line2D([0], [0], color="green", ls="--", label="sample anchor"))

    # 실제 렌더링 높이에 맞춰 footer를 다시 조정한다.
    # 범례 항목/행 수가 바뀌어도 그래프와 겹치지 않는다.
    HEADER_IN = 1.0
    FOOTER_IN = 0.75
    fig_h = 2.0*n_plots + HEADER_IN + FOOTER_IN
    fig = plt.figure(figsize=(22, fig_h))
    gs  = fig.add_gridspec(n_plots, 2, hspace=0.55, wspace=0.08, width_ratios=[1, 1.2],
                           top=1 - HEADER_IN/fig_h, bottom=FOOTER_IN/fig_h,
                           left=0.045, right=0.995)

    for r, info in enumerate(rows):
        aid = info["AnalysisID"]
        bid = aid_to_bam[aid]
        raw = get_vals(bid)
        m, a_kind = robust_anchor(raw)
        anchored  = raw - m
        z         = (anchored - pon_median) / pon_mad
        z_smooth  = rolling_median(z)

        n_up = int((z_smooth >  Z_CALL_THR).sum())
        n_dn = int((z_smooth < -Z_CALL_THR).sum())

        segs_base = extract_segments(z_smooth, bin_chrom=chrs_sorted)
        nf, bg_mad, segs = adaptive_noise_factor_and_segs(
            z, z_smooth, segs_base, chrs_sorted,
            tiers=SEG_TIERS, fallback_mad=info["bin_z_MAD"])
        bin_z_mad = float(stats.median_abs_deviation(z[np.isfinite(z)], nan_policy="omit"))
        max_peak_abs_z = float(np.nanmax(np.abs(z_smooth))) if np.any(np.isfinite(z_smooth)) else 0.0
        segs = augment_cfdna_diluted_segments(
            segs, z_smooth, nf, bin_z_mad=bin_z_mad,
            max_peak_abs_z=max_peak_abs_z,
            noisy_bin_mad=NOISY_BIN_MAD, bin_chrom=chrs_sorted, cfg=CFG)
        vis_segs = [s for s in segs
                    if seg_plot_tier(s, nf, bg_mad=bg_mad, nf_cap=NOISE_FACTOR_MAX,
                                     bin_z_mad=bin_z_mad, max_peak_abs_z=max_peak_abs_z,
                                     noisy_bin_mad=NOISY_BIN_MAD) is not None]

        # SEG/ARM 카운트 = sample_labels TSV 값 (라벨링에 쓰인 메트릭) 그대로
        label_str    = str(info["label"])
        n_focal_t    = int(info.get("n_seg_focal",     0) or 0)
        n_std_t      = int(info.get("n_seg_standard",  0) or 0)
        n_long_t     = int(info.get("n_seg_long",      0) or 0)
        n_vis_seg    = len(vis_segs)
        n_pos_arms   = int(info.get("n_positive_arms", 0) or 0)
        max_arm_mean = float(info.get("max_arm_abs_mean_z", 0.0) or 0.0)
        arm_list_raw = info.get("arm_positive_list", "")
        arm_list_str = "" if pd.isna(arm_list_raw) else str(arm_list_raw)
        if len(arm_list_str) > 95:
            arm_list_str = arm_list_str[:92] + "..."

        sex_str = {1: "M", 2: "F"}.get(int(info["Sex"]) if pd.notna(info["Sex"]) else 0, "?")
        age_str = f"{int(info['Age'])}" if pd.notna(info["Age"]) else "?"
        bg_mad_val = float(info.get("bg_bin_MAD", 0.0) or 0.0)
        title_str = (f"{aid}  |  Age={age_str:>3} {sex_str}  |  {info['CancerType']}/{info['Stage']}  |  "
                     f"Label={label_str:11s} (TSV)  |  "
                     f"gain={n_up:>4} loss={n_dn:>4}  |  SEG[f={n_focal_t}/s={n_std_t}/l={n_long_t} vis={n_vis_seg}]  "
                     f"n_pos_arms={n_pos_arms}  max_arm_|m|={max_arm_mean:.2f}  "
                     f"bin_MAD={info['bin_z_MAD']:.2f} bg_MAD={bg_mad_val:.2f} noise_x{nf:.2f}"
                     f"  |  anchor={m:+.4f} ({a_kind})\narms=[{arm_list_str}]")
        title_color = LABEL_TITLE_COLOR.get(label_str, "#1a9850")

        ax1 = fig.add_subplot(gs[r, 0])
        ax1.fill_between(x, raw_q5, raw_q95, color="#dddddd", alpha=0.9, zorder=1)
        ax1.scatter(x, raw, c="black", s=1.2, alpha=0.5, linewidths=0, zorder=2)
        ax1.axhline(0, color="black", lw=0.5)
        ax1.axhline(m, color="green", lw=0.8, ls="--", label=f"anchor={m:+.4f}")
        for i, c in enumerate(CHR_ORDER):
            if c not in boundaries: continue
            x0, x1 = boundaries[c]
            if i % 2 == 0: ax1.axvspan(x0, x1, color="#f5f5f5", alpha=0.5, zorder=0)
            if r == 0:
                ax1.text((x0+x1)/2, ylim_raw[1]-0.08, c, ha="center", fontsize=6, color="#444")
        ax1.set_ylim(*ylim_raw); ax1.set_xlim(-50, len(x)+50)
        ax1.set_ylabel("raw log2 CNR", fontsize=8)
        ax1.set_title(title_str, fontsize=8, fontweight="bold", color=title_color, loc="left")
        ax1.legend(fontsize=6, loc="lower right")
        if r == n_plots-1: ax1.set_xlabel("Genomic bins (chr 1-22)", fontsize=8)
        ax1.tick_params(labelsize=6)

        ax2 = fig.add_subplot(gs[r, 1])
        ax2.fill_between(x, z_q5, z_q95, color="#dddddd", alpha=0.9, zorder=1)
        for s in segs:
            draw_seg_band(ax2, s, nf, bg_mad=bg_mad, nf_cap=NOISE_FACTOR_MAX,
                          bin_z_mad=bin_z_mad, max_peak_abs_z=max_peak_abs_z,
                          noisy_bin_mad=NOISY_BIN_MAD)
        up = z_smooth >  Z_CALL_THR
        dn = z_smooth < -Z_CALL_THR
        nm = ~(up | dn)
        ax2.scatter(x[nm], z_smooth[nm], c="black",   s=1.3, alpha=0.5, linewidths=0, zorder=2)
        ax2.scatter(x[up], z_smooth[up], c="#d73027", s=2.3, alpha=0.85, linewidths=0, zorder=3)
        ax2.scatter(x[dn], z_smooth[dn], c="#2166ac", s=2.3, alpha=0.85, linewidths=0, zorder=3)
        ax2.axhline( 0, color="black",   lw=0.5)
        ax2.axhline( Z_CALL_THR, color="#2166ac", lw=0.4, ls="--", alpha=0.5)
        ax2.axhline(-Z_CALL_THR, color="#d73027", lw=0.4, ls="--", alpha=0.5)
        for i, c in enumerate(CHR_ORDER):
            if c not in boundaries: continue
            x0, x1 = boundaries[c]
            if i % 2 == 0: ax2.axvspan(x0, x1, color="#f5f5f5", alpha=0.5, zorder=0)
            if r == 0:
                ax2.text((x0+x1)/2, ylim_z[1]-4, c, ha="center", fontsize=6, color="#444")
        ax2.set_ylim(*ylim_z); ax2.set_xlim(-50, len(x)+50)
        ax2.set_ylabel("z (anchored, smoothed)", fontsize=8)
        if r == n_plots-1: ax2.set_xlabel("Genomic bins (chr 1-22)", fontsize=8)
        ax2.tick_params(labelsize=6)

    suptitle = fig.suptitle(
        f"single-sample CNV profile [depth={DEPTH_LABEL}; "
        f"PoN=healthy Train n={pon_display_n}; "
        f"Label/SEG/ARM = sample_labels_{DEPTH_LABEL}.tsv]",
        fontsize=12, fontweight="bold", y=1 - 0.22/fig_h)

    legend = fig.legend(handles=legend_elements, loc="lower left", fontsize=8,
                        ncol=LEG_NCOL, framealpha=0.9)
    fig.canvas.draw()
    legend_h_in = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    BOTTOM_PAD_IN = 0.10
    XLABEL_CLEARANCE_IN = 0.42
    FOOTER_IN = BOTTOM_PAD_IN + legend_h_in + XLABEL_CLEARANCE_IN
    fig_h = 2.0*n_plots + HEADER_IN + FOOTER_IN
    fig.set_size_inches(22, fig_h, forward=True)
    gs.update(top=1 - HEADER_IN/fig_h, bottom=FOOTER_IN/fig_h)
    legend.set_bbox_to_anchor((0.045, BOTTOM_PAD_IN/fig_h), transform=fig.transFigure)
    suptitle.set_y(1 - 0.22/fig_h)

    outdir = Path(args.outdir) if args.outdir else (OUT_DIR / "oneoff_profiles")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(aids) if len(aids) <= 3 else f"{aids[0]}_and_{len(aids)-1}more"
    out_path = outdir / f"{stem}_{DEPTH_LABEL}_{args.suffix}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
