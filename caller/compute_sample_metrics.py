#!/usr/bin/env python3
"""
compute_sample_metrics.py
전체 샘플별 CNV signal / noise metric 계산 (Step 1–9)
  - bin_z_MAD    : bin-level z-score MAD (raw noise + signal)
  - smooth_z_MAD : rolling_median(z, w=11) MAD (signal after smoothing)
  - coherence    : smooth_z_MAD / bin_z_MAD  (0~1, CNV+이면 큼)
  - n_seg_focal/standard/long, n_arms_strong/wide, noise_factor, ...

PoN     : healthy 중 split == "Train" 만 (config.splits.primary_split_col)
anchor  : combined - mode if |mode-median| < ANCHOR_TOL else median

설정: config.yaml (CFDNA_CNV_CONFIG 환경변수로 경로 override 가능)
"""

import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks

# caller 패키지 안에서 실행되든 script 로 실행되든 모두 동작하게
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config_loader import load_config, get_path  # noqa: E402
from utils.chr_arm_coords import ARM_COORDS, AUTOSOME_CHROMS, make_bin_arm  # noqa: E402
from segment_utils import (  # noqa: E402
    noise_factor_capped, summarize_weak_segments, NOISE_FACTOR_MAX_DEFAULT,
)
from cfdna_utils import cfdna_diluted_band_segments  # noqa: E402
from arm_utils import (  # noqa: E402
    CFDNA_DILUTED_CONC_TIERS as _CFDNA_DILUTED_CONC_TIERS_DEFAULT,
    CFDNA_CONC_Z3_THR as _CFDNA_CONC_Z3_THR_DEFAULT,
    diluted_bands_arm_concordant,
)

CFG = load_config()
print(f"[config] {CFG['_config_path']}")

WORK_DIR    = Path(CFG["paths"]["work_dir"])
QDNASEQ_MATRIX = get_path(CFG, "qdnaseq_matrix")
PON_REFERENCE = (Path(CFG["paths"]["pon_reference"])
                 if CFG["paths"].get("pon_reference") else None)
META_XLSX   = get_path(CFG, "meta_xlsx")
META_SHEET  = CFG["paths"].get("meta_sheet", "전체")
OUT_DIR     = Path(CFG["paths"].get("out_dir",
                                    WORK_DIR / "feature_matrix" / "batch_check" / "cnv_burden"))
DEPTH_LABEL = CFG.get("depth_label", "100G")
SPLIT_COL   = CFG.get("splits", {}).get("primary_split_col", "Split_AgeSexStage")

CHR_ORDER   = [str(i) for i in range(1, 23)]
WINDOW      = int(CFG.get("caller", {}).get("smooth_window", 11))
ANCHOR_TOL  = float(CFG.get("caller", {}).get("anchor_tol", 0.02))

_CALLER_CFG = CFG.get("caller", {})
# segment / arm 메트릭 파라미터
Z_CALL_THR        = float(_CALLER_CFG.get("z_call_thr",   3.0))   # 강한 outlier (bin 카운트 / arm_frac / segment hysteresis 시작 임계값)
Z_LO_THR          = float(_CALLER_CFG.get("z_lo_thr",     1.5))   # segment hysteresis 확장 임계값 (같은 부호 유지)
SEG_GAP_MAX       = int(  _CALLER_CFG.get("seg_gap_max",  20))    # hysteresis 중 lo_thr도 못 넘는 bin 허용 개수 (~2 Mb @ 100kb bin)
MIN_BINS_PER_ARM  = int(  _CALLER_CFG.get("min_bins_per_arm", 30))
# diluted band concordance: arm-mean frac is counted at this weak z threshold
# (matches the diluted hi_thr=1.0 — frac at z=3.0 is ~0 for weak-but-broad signals)
CFDNA_ARM_Z_CALL_THR = float(_CALLER_CFG.get("cfdna_arm_z_call_thr", 1.0))
CFDNA_BAND_ARM_OVERLAP_MIN = int(_CALLER_CFG.get("cfdna_band_arm_overlap_min", 40))
# Diluted-band ARM-concordance tier — exposed for depth-specific tuning.
# Defaults preserve the 100G baseline (see caller/arm_utils.py:29-31, 36).
_conc_default = _CFDNA_DILUTED_CONC_TIERS_DEFAULT[0]
CFDNA_CONC_MEAN_MIN     = float(_CALLER_CFG.get("cfdna_conc_mean_min",     _conc_default["mean_min"]))
CFDNA_CONC_FRAC_MIN     = float(_CALLER_CFG.get("cfdna_conc_frac_min",     _conc_default["frac_min"]))
CFDNA_CONC_FRAC_Z3_MIN  = float(_CALLER_CFG.get("cfdna_conc_frac_z3_min",  _conc_default["frac_z3_min"]))
CFDNA_CONC_Z3_THR       = float(_CALLER_CFG.get("cfdna_conc_z3_thr",       _CFDNA_CONC_Z3_THR_DEFAULT))
CFDNA_DILUTED_CONC_TIERS = [{
    "name":         "conc",
    "mean_min":     CFDNA_CONC_MEAN_MIN,
    "frac_min":     CFDNA_CONC_FRAC_MIN,
    "frac_z3_min":  CFDNA_CONC_FRAC_Z3_MIN,
}]

# Segment 다층 tier (길이-진폭 trade-off):
#   focal    : 짧고 매우 강한 신호 (focal amplification/deletion)
#   standard : 중간 길이 + 중간 강도 (기존 기본)
#   long     : 길지만 약한 신호 (sub-arm 전반 shift)
SEG_TIERS = [
    # focal: 0.5~3Mb 짧은 고강도. noise scaling 면제 (raw absolute threshold) +
    # 이중 SNR 보호 (mean/bg, peak/bg) 으로 noisy 우연 spike 차단.
    {"name": "focal",    "L_min": 5,   "L_max": 30,  "amp_min": 7.0, "peak_min": 10.0,
     "density_min": 0.50, "snr_min": 5.0, "peak_snr_min": 7.0},
    # standard/long 은 background-MAD 기반 noise scaling 대상.
    {"name": "standard", "L_min": 30,  "amp_min": 4.5, "peak_min": 5.0,  "density_min": 0.30},
    {"name": "long",     "L_min": 300, "amp_min": 2.0, "peak_min": 3.0,  "density_min": 0.10},
]

# focal subsegment 추출 파라미터: hysteresis 가 noisy bin 과 합쳐 length>L_max 가 된
# 큰 segment 안에서, peak 위치 중심으로 same-direction & |z|>=hi_abs 연속 bin 확장 후
# focal 조건 (amp/peak/density/SNR) 평가.
FOCAL_SUBSEG_HI_ABS = float(_CALLER_CFG.get("focal_subseg_hi_abs", 2.0))   # 2 로 낮추면 noisy 주변에서 sub-window 가 더 잘 확장

# Arm 다층 tier:
#   strong : arm 평균이 크게 shift
#   wide   : 평균 shift는 작지만 같은 방향 bin 비율이 매우 높음 (continuity 위주)
ARM_TIERS = [
    {"name": "strong", "mean_min": 2.5, "frac_min": 0.30},
    {"name": "wide",   "mean_min": 1.2, "frac_min": 0.50},
]

# Sample-level noise-adaptive scaling (background-MAD 기반 2-pass):
#   Pass 0  : raw tier (scaling 없음)로 segment/arm 추출 -> 유의 segment 영역 확보
#   Pass 1  : 유의 segment bin 제외한 영역(=background) 에서 bin-level MAD 재계산
#             -> nf_p1 = max(1, bg_mad_p1 / NOISE_REF_MAD)
#             -> scaled tier 로 segment/arm 재평가
#   Pass 2  : Pass 1 의 새 유의 segment 영역 제외하고 background MAD 재계산
#             -> nf_p2 = max(1, bg_mad_p2 / NOISE_REF_MAD)  (수렴 가까움)
#             -> 최종 tier 평가
#   이렇게 하면 cancer 처럼 신호가 광범위해 bin_z_MAD 가 inflate 되는 케이스도
#   신호 영역을 제외한 진짜 background 기준으로 threshold 가 결정됨.
NOISE_REF_MAD       = float(_CALLER_CFG.get("noise_ref_mad",    1.0))   # Reference scale for noise adjustment
NOISE_FACTOR_MAX    = float(_CALLER_CFG.get("noise_factor_max", NOISE_FACTOR_MAX_DEFAULT))
SEG_SCALE_KEYS      = ("amp_min", "peak_min")
ARM_SCALE_KEYS      = ("mean_min",)
BG_MIN_BIN_COUNT    = int(_CALLER_CFG.get("bg_min_bin_count", 2000))  # background bin 이 부족하면 전체 z 의 MAD 로 fallback


def noise_factor(bin_z_mad, ref_mad=NOISE_REF_MAD):
    if not np.isfinite(bin_z_mad) or bin_z_mad <= ref_mad:
        return 1.0
    return float(bin_z_mad / ref_mad)


def make_scaled_tiers(base_tiers, factor, scale_keys):
    """tier 의 amplitude 계열 임계값을 factor 배로 상향한 새 tier list.

    focal tier 는 noise scaling 면제 — focal 신호는 sample-wide noise 와 분리되며
    이미 충분히 strict (amp>=7, peak>=10, density>=0.5, L 15~30 bin).
    """
    if factor == 1.0:
        return base_tiers
    out = []
    for t in base_tiers:
        new_t = dict(t)
        if t["name"] != "focal":
            for k in scale_keys:
                if k in new_t:
                    new_t[k] = new_t[k] * factor
        out.append(new_t)
    return out


def _seg_passes_tier(seg, tier, noise_factor=1.0, bg_mad=None):
    """L_max, nf-scaled threshold, snr_min 모두 통과 검사.

    snr_min 은 mean_abs_z / bg_mad. bg_mad 가 None 이거나 0 이면 SNR 평가 skip.
    """
    if seg["length"] < tier["L_min"]:
        return False
    if "L_max" in tier and seg["length"] > tier["L_max"]:
        return False
    if not (seg["mean_abs_z"] >= tier["amp_min"]  * noise_factor
            and seg["peak_abs_z"] >= tier["peak_min"] * noise_factor
            and seg["density"]    >= tier["density_min"]):
        return False
    if bg_mad is not None and bg_mad > 0:
        snr_min = tier.get("snr_min")
        if snr_min is not None and seg["mean_abs_z"] / bg_mad < snr_min:
            return False
        peak_snr_min = tier.get("peak_snr_min")
        if peak_snr_min is not None and seg["peak_abs_z"] / bg_mad < peak_snr_min:
            return False
    return True


def significant_seg_mask(segments, n_bins, tiers, noise_factor=1.0, bg_mad=None):
    """tier 통과 segment 의 bin 위치를 True 로 표시한 boolean mask."""
    mask = np.zeros(n_bins, dtype=bool)
    for s in segments:
        for t in tiers:
            if _seg_passes_tier(s, t, noise_factor=noise_factor, bg_mad=bg_mad):
                mask[s["start"]:s["end"]+1] = True
                break
    return mask


def extract_focal_subsegments(segments, z_smooth, bin_chrom,
                               focal_tier, bg_mad,
                               hi_abs=FOCAL_SUBSEG_HI_ABS):
    """base segment 안에서 peak 중심 focal 후보 sub-window 를 추출.

    대상:
      1) length > L_max 인 큰 segment (hysteresis 가 noisy bin 과 합친 케이스)
      2) length ≤ L_max 이지만 base 에서 focal 통과 못 한 segment
         (예: density 미달 — peak 근처만 dense 한 sparse 신호)
    각 segment 마다 1 개의 sub-focal 후보만 (peak 위치 기준) 추출.
    """
    L_min = focal_tier["L_min"]
    L_max = focal_tier["L_max"]
    out = []
    for s in segments:
        if s["length"] <= L_max:
            # base 에서 이미 focal 통과 → 중복 방지로 skip
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

        # peak 중심으로 양쪽으로 same-direction & |z|>=hi_abs 인 한 확장 (L_max 까지)
        l = peak_local
        while l > 0:
            cand = sub[l-1]
            if not np.isfinite(cand): break
            if chrom_sub is not None and chrom_sub[l-1] != peak_chrom: break
            if direction * cand < hi_abs: break
            if (peak_local - (l-1) + 1) > L_max: break
            l -= 1
        r = peak_local
        while r < len(sub) - 1:
            cand = sub[r+1]
            if not np.isfinite(cand): break
            if chrom_sub is not None and chrom_sub[r+1] != peak_chrom: break
            if direction * cand < hi_abs: break
            if ((r+1) - l + 1) > L_max: break
            r += 1
        win = sub[l:r+1]
        win_f = win[np.isfinite(win)]
        win_len = r - l + 1
        if win_len < L_min:
            continue
        abs_w = np.abs(win_f)
        mean_a = float(abs_w.mean())
        peak_a = float(abs_w.max())
        n_strict = int((direction * win_f >= hi_abs).sum())
        density  = n_strict / win_len if win_len > 0 else 0.0
        cand_seg = {
            "start":      int(s["start"] + l),
            "end":        int(s["start"] + r),
            "direction":  int(direction),
            "length":     int(win_len),
            "mean_abs_z": mean_a,
            "peak_abs_z": peak_a,
            "n_strict":   n_strict,
            "density":    float(density),
            "from_subseg": True,
        }
        if _seg_passes_tier(cand_seg, focal_tier, noise_factor=1.0, bg_mad=bg_mad):
            out.append(cand_seg)
    return out


def background_bin_mad(z_bin, sig_mask, fallback_mad,
                       min_bin_count=BG_MIN_BIN_COUNT):
    """유의 segment 영역 제외한 bin-level z 의 MAD."""
    bg = z_bin[~sig_mask]
    bg = bg[np.isfinite(bg)]
    if len(bg) < min_bin_count:
        return float(fallback_mad)
    return float(stats.median_abs_deviation(bg, nan_policy="omit"))

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("meta 로드...")
_meta_cols = ["AnalysisID", "CancerType", "Stage"]
for _c in ("Split_AgeSexStage", "Split_ClinicalStrict", "Split_ClinicalStrict_ASCprefix"):
    if _c not in _meta_cols:
        _meta_cols.append(_c)
meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)
_have = [c for c in _meta_cols if c in meta.columns]
meta = meta[_have]
train_healthy_aids = set(meta[(meta["CancerType"]=="healthy") &
                              (meta[SPLIT_COL]=="Train")]["AnalysisID"])
print(f"  Excel healthy Train: {len(train_healthy_aids)}")

print(f"{DEPTH_LABEL} matrix header 로드 ({QDNASEQ_MATRIX})...")
with open(QDNASEQ_MATRIX) as f:
    header = f.readline().strip().split("\t")
bam_ids = [h for h in header if h != "feature"]
meta_aids = set(meta["AnalysisID"].dropna().astype(str))
aid_to_bam = {}
for b in bam_ids:
    parts = b.split("_")
    pid = parts[0]
    if pid in meta_aids:
        aid_to_bam[pid] = b
    else:
        for n in range(2, len(parts) + 1):
            candidate = "_".join(parts[:n])
            if candidate in meta_aids:
                aid_to_bam[candidate] = b
                break
bam_to_aid = {v:k for k,v in aid_to_bam.items()}
keep_bam_ids = list(aid_to_bam.values())
print(f"  matrix samples: {len(bam_ids)}  (AnalysisID-mapped kept: {len(keep_bam_ids)})")

print(f"{DEPTH_LABEL} matrix 로드...")
mat = pd.read_csv(QDNASEQ_MATRIX, sep="\t", index_col=0)
mat = mat[~mat.index.str.startswith("X:")]
print(f"  shape: {mat.shape}")

chrs   = mat.index.str.extract(r"^(\w+):")[0]
starts = mat.index.str.extract(r":(\d+)-")[0].astype(int)
mask   = chrs.isin(CHR_ORDER)
chr_cat  = pd.Categorical(chrs[mask], categories=CHR_ORDER, ordered=True)
sort_idx = np.lexsort([starts.values[mask], chr_cat.codes])

print("autosome only + chr 정렬...")
mat_arr = mat.values[mask][sort_idx].astype(np.float32)
features_sorted = mat.index.values[mask][sort_idx].astype(str)
bam_cols = list(mat.columns)
del mat
print(f"  arr shape: {mat_arr.shape}")

chrs_sorted   = chrs[mask].values[sort_idx]
starts_sorted = starts.values[mask][sort_idx]


def fast_mode(v, lo=-0.3, hi=0.3, bw=0.005):
    v2 = v[np.isfinite(v) & (v>lo) & (v<hi)]
    if len(v2) < 100: return 0.0
    edges = np.arange(lo, hi+bw, bw)
    h, _  = np.histogram(v2, bins=edges)
    centers = (edges[:-1]+edges[1:])/2
    h_s = np.convolve(h, np.ones(3)/3, mode="same")
    peaks, _ = find_peaks(h_s, height=h_s.max()*0.5)
    if len(peaks)==0: return float(centers[np.argmax(h_s)])
    return float(centers[peaks[np.argmax(h_s[peaks])]])

def robust_anchor(v):
    mode_v   = fast_mode(v)
    median_v = float(np.nanmedian(v))
    if abs(mode_v - median_v) < ANCHOR_TOL:
        return mode_v, "mode"
    return median_v, "median"

def rolling_median(v, w=WINDOW):
    return pd.Series(v).rolling(window=w, center=True, min_periods=w//2).median().values


def build_bin_to_arm(chrs_sorted, starts_sorted, arm_coords, autosome_chroms):
    """sorted bin index -> arm index 매핑. autosome 외/centromere 영역은 -1."""
    autosome_set = set(autosome_chroms)
    arm_idx_of_arm = {ac.arm: i for i, ac in enumerate(arm_coords) if ac.chrom in autosome_set}
    chrom_arms = {}
    for ac in arm_coords:
        if ac.chrom not in autosome_set:
            continue
        chrom_arms.setdefault(ac.chrom, []).append(ac)

    n = len(chrs_sorted)
    bin_to_arm = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        chrom_key = f"chr{chrs_sorted[i]}"
        arms = chrom_arms.get(chrom_key)
        if arms is None:
            continue
        s = starts_sorted[i]
        for ac in arms:
            if ac.start <= s < ac.end:
                bin_to_arm[i] = arm_idx_of_arm[ac.arm]
                break
    return bin_to_arm


def extract_segments(z_smooth, bin_chrom=None, bin_arm=None,
                     hi_thr=Z_CALL_THR, lo_thr=Z_LO_THR, gap_max=SEG_GAP_MAX):
    """Hysteresis (Schmitt-trigger) 기반 segment 추출.

    상태 머신:
      outside (0):
        z >= +hi_thr -> +1 상태 진입 (seg 시작)
        z <= -hi_thr -> -1 상태 진입
      in_gain (+1):
        z >= +lo_thr -> 확장 (last_extend=i, gap=0)
        z <= -hi_thr -> seg 종료, 즉시 -1 시작
        그 외        -> gap_counter++; > gap_max 이면 종료
      in_loss (-1): 대칭

    bin_chrom 가 주어지면 chromosome 경계에서 강제 종료.
    bin_arm 가 주어지면 p/q arm 경계에서도 강제 종료 (centromere bin = None 은 gap 취급).
    NaN/Inf 는 gap 으로 취급.

    반환: list of {start, end, direction, length, mean_abs_z, peak_abs_z, density, n_strict}.
    """
    n = len(z_smooth)
    finite = np.isfinite(z_smooth)
    segments = []

    state = 0
    seg_start = -1
    last_extend = -1
    gap_counter = 0
    prev_chrom = None
    prev_arm = None

    def _close(direction):
        if seg_start < 0 or last_extend < 0 or last_extend < seg_start:
            return
        sub = z_smooth[seg_start:last_extend + 1]
        sub_f = sub[np.isfinite(sub)]
        if len(sub_f) == 0:
            return
        abs_v = np.abs(sub_f)
        if direction > 0:
            n_strict = int((sub_f >= hi_thr).sum())
        else:
            n_strict = int((sub_f <= -hi_thr).sum())
        length = int(last_extend - seg_start + 1)
        segments.append({
            "start":      int(seg_start),
            "end":        int(last_extend),
            "direction":  int(direction),
            "length":     length,
            "mean_abs_z": float(abs_v.mean()),
            "peak_abs_z": float(abs_v.max()),
            "n_strict":   n_strict,
            "density":    float(n_strict / length) if length > 0 else 0.0,
        })

    for i in range(n):
        cur_chrom = bin_chrom[i] if bin_chrom is not None else None
        cur_arm   = bin_arm[i]   if bin_arm   is not None else None

        # chromosome 경계: 강제 종료
        if state != 0 and prev_chrom is not None and cur_chrom is not None and cur_chrom != prev_chrom:
            _close(state)
            state = 0
            seg_start = last_extend = -1
            gap_counter = 0

        # p/q arm 경계: 강제 종료 (centromere 진입/탈출 포함, arm label 변화 시 무조건 종료)
        elif state != 0 and cur_arm != prev_arm:
            _close(state)
            state = 0
            seg_start = last_extend = -1
            gap_counter = 0

        if not finite[i]:
            if state != 0:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state)
                    state = 0
                    seg_start = last_extend = -1
                    gap_counter = 0
            prev_chrom = cur_chrom
            prev_arm = cur_arm
            continue

        zv = z_smooth[i]
        if state == 0:
            if zv >= hi_thr:
                state = 1; seg_start = last_extend = i; gap_counter = 0
            elif zv <= -hi_thr:
                state = -1; seg_start = last_extend = i; gap_counter = 0
        elif state == 1:
            if zv <= -hi_thr:
                _close(state)
                state = -1; seg_start = last_extend = i; gap_counter = 0
            elif zv >= lo_thr:
                last_extend = i; gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state)
                    state = 0; seg_start = last_extend = -1; gap_counter = 0
        else:  # state == -1
            if zv >= hi_thr:
                _close(state)
                state = 1; seg_start = last_extend = i; gap_counter = 0
            elif zv <= -lo_thr:
                last_extend = i; gap_counter = 0
            else:
                gap_counter += 1
                if gap_counter > gap_max:
                    _close(state)
                    state = 0; seg_start = last_extend = -1; gap_counter = 0

        prev_chrom = cur_chrom
        prev_arm = cur_arm

    if state != 0:
        _close(state)

    return segments


def _seg_tier_mask(lens, amps, peaks, dens, tier, bg_mad=None):
    """Tier 조건 모두 만족하는 segment의 boolean mask. L_max, snr_min 도 적용."""
    m = ((lens >= tier["L_min"]) & (amps >= tier["amp_min"])
         & (peaks >= tier["peak_min"]) & (dens >= tier["density_min"]))
    if "L_max" in tier:
        m &= (lens <= tier["L_max"])
    if bg_mad is not None and bg_mad > 0:
        snr_min = tier.get("snr_min")
        if snr_min is not None:
            m &= ((amps / bg_mad) >= snr_min)
        peak_snr_min = tier.get("peak_snr_min")
        if peak_snr_min is not None:
            m &= ((peaks / bg_mad) >= peak_snr_min)
    return m


def summarize_segments(segments, tiers=SEG_TIERS, bg_mad=None):
    """segment 리스트 -> sample-level 통계 dict.

    유의 segment = SEG_TIERS 중 어느 한 tier라도 만족.
    """
    n_total = len(segments)
    out = {
        "n_seg_total":           int(n_total),
        "n_seg_significant":     0,
        "max_seg_len":           0,
        "max_seg_mean_abs_z":    0.0,
        "max_seg_peak_abs_z":    0.0,
        "max_seg_density":       0.0,
        "total_signif_seg_bins": 0,
    }
    for t in tiers:
        out[f"n_seg_{t['name']}"] = 0
    if n_total == 0:
        return out
    lens  = np.array([s["length"]     for s in segments])
    amps  = np.array([s["mean_abs_z"] for s in segments])
    peaks = np.array([s["peak_abs_z"] for s in segments])
    dens  = np.array([s["density"]    for s in segments])
    any_mask = np.zeros(n_total, dtype=bool)
    for t in tiers:
        m = _seg_tier_mask(lens, amps, peaks, dens, t, bg_mad=bg_mad)
        out[f"n_seg_{t['name']}"] = int(m.sum())
        any_mask |= m
    out["n_seg_significant"]     = int(any_mask.sum())
    out["max_seg_len"]           = int(lens.max())
    out["max_seg_mean_abs_z"]    = float(amps.max())
    out["max_seg_peak_abs_z"]    = float(peaks.max())
    out["max_seg_density"]       = float(dens.max())
    out["total_signif_seg_bins"] = int(lens[any_mask].sum())
    return out


def is_segment_significant(seg, tiers=SEG_TIERS):
    """단일 segment dict -> 어느 tier라도 통과하면 True."""
    for t in tiers:
        if (seg["length"]     >= t["L_min"]
            and seg["mean_abs_z"] >= t["amp_min"]
            and seg["peak_abs_z"] >= t["peak_min"]
            and seg["density"]    >= t["density_min"]):
            return True
    return False


def compute_arm_stats(z_smooth, bin_to_arm, n_arms, z_call_thr=Z_CALL_THR,
                       min_bins_per_arm=MIN_BINS_PER_ARM):
    """arm별 (mean_z, frac_pos, frac_neg, n_bins) 계산. n_bins < 임계면 NaN."""
    out = {
        "mean_z":   np.full(n_arms, np.nan),
        "frac_pos": np.full(n_arms, np.nan),
        "frac_neg": np.full(n_arms, np.nan),
        "n_bins":   np.zeros(n_arms, dtype=np.int32),
    }
    for a in range(n_arms):
        sel = bin_to_arm == a
        if not sel.any():
            continue
        z = z_smooth[sel]
        z = z[np.isfinite(z)]
        n = len(z)
        out["n_bins"][a] = n
        if n < min_bins_per_arm:
            continue
        out["mean_z"][a]   = float(np.mean(z))
        out["frac_pos"][a] = float(np.mean(z >  z_call_thr))
        out["frac_neg"][a] = float(np.mean(z < -z_call_thr))
    return out


def summarize_arms(arm_stats, arm_names, tiers=ARM_TIERS):
    """arm 통계 -> sample-level dict.

    유의 arm = ARM_TIERS 중 어느 한 tier라도 만족.
    """
    mean_z   = arm_stats["mean_z"]
    frac_pos = arm_stats["frac_pos"]
    frac_neg = arm_stats["frac_neg"]
    finite   = np.isfinite(mean_z)
    out = {
        "n_positive_arms":      0,
        "max_arm_abs_mean_z":   0.0,
        "max_arm_frac_outlier": 0.0,
        "arm_positive_list":    "",
    }
    for t in tiers:
        out[f"n_arms_{t['name']}"] = 0
    if not finite.any():
        return out
    abs_mean  = np.where(finite, np.abs(mean_z), 0.0)
    frac_any  = np.where(finite, np.maximum(np.nan_to_num(frac_pos), np.nan_to_num(frac_neg)), 0.0)
    any_mask = np.zeros(len(mean_z), dtype=bool)
    for t in tiers:
        m = finite & (abs_mean >= t["mean_min"]) & (frac_any >= t["frac_min"])
        out[f"n_arms_{t['name']}"] = int(m.sum())
        any_mask |= m
    pos_arms = []
    for i in np.where(any_mask)[0]:
        sign = "+" if mean_z[i] > 0 else "-"
        pos_arms.append(f"{arm_names[i]}{sign}")
    out["n_positive_arms"]      = int(any_mask.sum())
    out["max_arm_abs_mean_z"]   = float(abs_mean.max())
    out["max_arm_frac_outlier"] = float(frac_any.max())
    out["arm_positive_list"]    = ",".join(pos_arms)
    return out


print("bin -> arm 매핑 계산...")
arm_names    = [ac.arm for ac in ARM_COORDS if ac.chrom in set(AUTOSOME_CHROMS)]
arm_coords_a = [ac     for ac in ARM_COORDS if ac.chrom in set(AUTOSOME_CHROMS)]
bin_to_arm   = build_bin_to_arm(chrs_sorted, starts_sorted, arm_coords_a, AUTOSOME_CHROMS)
n_arms       = len(arm_names)
arm_bin_counts = np.array([(bin_to_arm == i).sum() for i in range(n_arms)])
print(f"  arms: {n_arms}, bin 수 min={arm_bin_counts.min()} max={arm_bin_counts.max()} "
      f"(MIN_BINS_PER_ARM={MIN_BINS_PER_ARM} 이하 arm: "
      f"{int((arm_bin_counts<MIN_BINS_PER_ARM).sum())}개)")
bin_arm = make_bin_arm(chrs_sorted, starts_sorted)

if PON_REFERENCE is not None:
    print(f"\nPoN reference 로드: {PON_REFERENCE}")
    if not PON_REFERENCE.is_file():
        raise SystemExit(f"PoN reference 없음: {PON_REFERENCE}")
    with np.load(PON_REFERENCE, allow_pickle=False) as ref:
        ref_depth = str(ref["depth_label"].item())
        ref_features = ref["features"].astype(str)
        pon_median = ref["pon_median"].astype(np.float32)
        pon_mad = ref["pon_mad"].astype(np.float32)
        pon_effective_n = int(ref["pon_n_effective"].item())
    if ref_depth != DEPTH_LABEL:
        raise SystemExit(
            f"PoN reference depth 불일치: config={DEPTH_LABEL}, reference={ref_depth}")
    if not np.array_equal(features_sorted, ref_features):
        raise SystemExit(
            "QDNAseq matrix feature/order가 PoN reference와 일치하지 않습니다.")
    print(f"  compact PoN loaded: effective n={pon_effective_n}, bins={len(pon_median)}")
else:
    print("\nPoN (healthy train) anchored stack 계산...")
    train_bam_ids = [aid_to_bam[a] for a in train_healthy_aids if a in aid_to_bam]
    train_col_idx = [bam_cols.index(b) for b in train_bam_ids]
    print(f"  {DEPTH_LABEL} ∩ train healthy: {len(train_bam_ids)}")

    if len(train_bam_ids) < 200:
        print(f"  [WARNING] PoN size ({len(train_bam_ids)}) < 200. MAD estimates may be unstable.")

    # ── 1-pass: anchor all train samples, compute raw bin MAD for noise filter ──
    PON_MAD_PCT_MAX = float(_CALLER_CFG.get("pon_mad_percentile_max", 90))
    train_anchored_all = np.empty((len(train_col_idx), mat_arr.shape[0]), dtype=np.float32)
    train_anchors_all  = np.empty(len(train_col_idx), dtype=np.float32)
    train_raw_mads     = np.empty(len(train_col_idx), dtype=np.float32)
    for i, ci in enumerate(train_col_idx):
        v = mat_arr[:, ci]
        a, _ = robust_anchor(v)
        anchored = v - a
        train_anchored_all[i] = anchored
        train_anchors_all[i]  = a
        finite = anchored[np.isfinite(anchored)]
        train_raw_mads[i] = stats.median_abs_deviation(
            finite, nan_policy="omit") if len(finite) > 0 else 0.0

    mad_thr = np.percentile(train_raw_mads, PON_MAD_PCT_MAX)
    clean_mask = train_raw_mads <= mad_thr
    n_excluded = int((~clean_mask).sum())
    print(f"  PoN noise filter: MAD thr={mad_thr:.4f} (Q{PON_MAD_PCT_MAX:.0f}), "
          f"excluded {n_excluded}/{len(train_col_idx)} noisy samples")

    # ── PoN 통계: clean samples만 사용 ──
    train_anchored = train_anchored_all[clean_mask]
    train_anchors  = train_anchors_all[clean_mask]
    print(f"  PoN effective size: {clean_mask.sum()}")

    pon_median = np.nanmedian(train_anchored, axis=0)
    pon_mad = stats.median_abs_deviation(
        train_anchored, axis=0, nan_policy="omit")
    pon_mad = np.where(pon_mad < 0.005, 0.005, pon_mad)
    print(f"  PoN median range: [{np.nanmin(pon_median):+.3f}, "
          f"{np.nanmax(pon_median):+.3f}]")
    print(f"  PoN MAD    range: [{np.nanmin(pon_mad):.3f}, "
          f"{np.nanmax(pon_mad):.3f}]")

print(f"\n모든 {DEPTH_LABEL} 샘플 metric 계산 (AnalysisID 매핑 샘플)...")
rows = []
keep_set = set(keep_bam_ids)
for i, bam in enumerate(bam_cols):
    if bam not in keep_set:
        continue
    aid = bam_to_aid[bam]
    v = mat_arr[:, i]
    if not np.isfinite(v).any():
        continue
    a_val, a_kind = robust_anchor(v)
    z       = (v - a_val - pon_median) / pon_mad
    z_smooth = rolling_median(z)

    finite      = np.isfinite(z)
    finite_sm   = np.isfinite(z_smooth)
    bin_mad     = stats.median_abs_deviation(z[finite],          nan_policy="omit")
    smooth_mad  = stats.median_abs_deviation(z_smooth[finite_sm], nan_policy="omit")
    coherence   = smooth_mad / bin_mad if bin_mad > 0 else 0.0
    n_gain_smooth = int((z_smooth >  Z_CALL_THR).sum())
    n_loss_smooth = int((z_smooth < -Z_CALL_THR).sum())

    segments_base = extract_segments(z_smooth, bin_chrom=chrs_sorted, bin_arm=bin_arm,
                                      hi_thr=Z_CALL_THR, lo_thr=Z_LO_THR, gap_max=SEG_GAP_MAX)
    arm_stats = compute_arm_stats(z_smooth, bin_to_arm, n_arms,
                                   z_call_thr=Z_CALL_THR,
                                   min_bins_per_arm=MIN_BINS_PER_ARM)
    n_bins = len(z)
    focal_tier_base = SEG_TIERS[0]  # name=="focal"

    # ── Pass 0: raw tier (scaling 없음) + bin_mad 로 임시 SNR — 첫 signal mask 추출 ──
    #   bg_mad 가 아직 없으므로 bin_mad 로 대체 (보수적). subsegments 도 임시 추출.
    sub_p0     = extract_focal_subsegments(segments_base, z_smooth, chrs_sorted,
                                            focal_tier_base, bg_mad=bin_mad)
    segs_p0    = segments_base + sub_p0
    sig_mask_p0 = significant_seg_mask(segs_p0, n_bins, SEG_TIERS,
                                        noise_factor=1.0, bg_mad=bin_mad)

    # ── Pass 1: background MAD 재계산 → focal subsegments 재추출 → scaled tier 평가 ──
    bg_mad_p1     = background_bin_mad(z, sig_mask_p0, fallback_mad=bin_mad)
    nf_p1         = noise_factor(bg_mad_p1)
    seg_tiers_p1  = make_scaled_tiers(SEG_TIERS, nf_p1, SEG_SCALE_KEYS)
    sub_p1        = extract_focal_subsegments(segments_base, z_smooth, chrs_sorted,
                                               focal_tier_base, bg_mad=bg_mad_p1)
    segs_p1       = segments_base + sub_p1
    sig_mask_p1   = significant_seg_mask(segs_p1, n_bins, seg_tiers_p1,
                                          noise_factor=1.0, bg_mad=bg_mad_p1)

    # ── Pass 2: 새 signal mask 로 background 재계산 → 최종 nf, 최종 metric ──
    bg_mad_p2     = background_bin_mad(z, sig_mask_p1, fallback_mad=bg_mad_p1)
    nf            = noise_factor(bg_mad_p2)
    seg_tiers_eff = make_scaled_tiers(SEG_TIERS, nf, SEG_SCALE_KEYS)
    arm_tiers_eff = make_scaled_tiers(ARM_TIERS, nf, ARM_SCALE_KEYS)
    sub_p2        = extract_focal_subsegments(segments_base, z_smooth, chrs_sorted,
                                               focal_tier_base, bg_mad=bg_mad_p2)
    segments      = segments_base + sub_p2
    seg_sum       = summarize_segments(segments, tiers=seg_tiers_eff, bg_mad=bg_mad_p2)
    arm_sum       = summarize_arms(arm_stats, arm_names, tiers=arm_tiers_eff)
    nf_capped     = noise_factor_capped(nf, cap=NOISE_FACTOR_MAX)

    # focal / significant segment chr.arm 위치 추출
    def _bin_arm_label(s):
        arm_i = int(bin_to_arm[s["start"]])
        if 0 <= arm_i < len(arm_names):
            return arm_names[arm_i] + ("+" if s["direction"] > 0 else "-")
        return None

    _focal_tier_eff = seg_tiers_eff[0]   # focal: noise-factor 면제
    _seen_focal = {}
    for s in segments:
        if _seg_passes_tier(s, _focal_tier_eff, noise_factor=1.0, bg_mad=bg_mad_p2):
            lbl = _bin_arm_label(s)
            if lbl and lbl not in _seen_focal:
                _seen_focal[lbl] = True
    focal_seg_arm_list = ",".join(_seen_focal)

    _seen_sig = {}
    for s in segments:
        for t in seg_tiers_eff[1:]:   # standard + long (noise scaling 내재됨)
            if _seg_passes_tier(s, t, noise_factor=1.0, bg_mad=bg_mad_p2):
                lbl = _bin_arm_label(s)
                if lbl and lbl not in _seen_sig:
                    _seen_sig[lbl] = True
                break
    sig_seg_arm_list = ",".join(_seen_sig)
    weak_sum      = summarize_weak_segments(segments, nf, bg_mad=bg_mad_p2,
                                            nf_cap=NOISE_FACTOR_MAX)

    # diluted band: focal subseg 제외한 peak max로 gate 판정
    nonfocal_peaks = [s["peak_abs_z"] for s in segments if not s.get("from_subseg")]
    max_peak_nonfocal = max(nonfocal_peaks, default=0.0)
    diluted_segs = cfdna_diluted_band_segments(
        segments, z_smooth, bin_mad, nf, max_peak_nonfocal,
        bin_chrom=chrs_sorted, cfg=CFG)
    n_diluted_gain = sum(1 for s in diluted_segs if s["direction"] > 0)
    n_diluted_loss = sum(1 for s in diluted_segs if s["direction"] < 0)

    # diluted band concordance: keep only GAIN bands corroborated by a same-direction
    # arm-mean shift passing CFDNA_DILUTED_CONC_TIERS. The tier checks three axes:
    #   - arm mean amplitude (mean_min) on z_smooth
    #   - weak-outlier density (frac_min) at z=CFDNA_ARM_Z_CALL_THR (e.g. 1.0)
    #   - strong-outlier density (frac_z3_min) at z=CFDNA_CONC_Z3_THR (=3.0)
    # frac at z=1 separates broad signal from focal; frac at z=3 separates real
    # low-fraction tumor (high strong-outlier density) from noisy healthy drift.
    cfdna_arm_stats = compute_arm_stats(
        z_smooth, bin_to_arm, n_arms,
        z_call_thr=CFDNA_ARM_Z_CALL_THR, min_bins_per_arm=MIN_BINS_PER_ARM)
    cfdna_arm_stats_z3 = compute_arm_stats(
        z_smooth, bin_to_arm, n_arms,
        z_call_thr=CFDNA_CONC_Z3_THR, min_bins_per_arm=MIN_BINS_PER_ARM)
    cfdna_arm_stats["frac_pos_z3"] = cfdna_arm_stats_z3["frac_pos"]
    cfdna_arm_stats["frac_neg_z3"] = cfdna_arm_stats_z3["frac_neg"]
    n_dil_conc_gain, n_dil_conc_loss, dil_conc_arms = diluted_bands_arm_concordant(
        diluted_segs, bin_arm, cfdna_arm_stats, arm_names,
        tiers=CFDNA_DILUTED_CONC_TIERS, min_overlap_bins=CFDNA_BAND_ARM_OVERLAP_MIN,
        gain_only=True)

    rows.append({
        "AnalysisID":            aid,
        "anchor":                a_val,
        "anchor_kind":           a_kind,
        "bin_z_MAD":             bin_mad,
        "smooth_z_MAD":          smooth_mad,
        "coherence":             coherence,
        "n_gain_smooth":         n_gain_smooth,
        "n_loss_smooth":         n_loss_smooth,
        "n_outlier_smooth":      n_gain_smooth + n_loss_smooth,
        # sample-level noise scaling (background-MAD 기반 2-pass)
        "noise_factor":          nf,            # 최종 적용된 factor (Pass 2)
        "noise_factor_capped":   nf_capped,     # min(nf, NOISE_FACTOR_MAX) — weak tier 용
        "noise_factor_p1":       nf_p1,         # Pass 1 factor (수렴 모니터링용)
        "bg_bin_MAD":            bg_mad_p2,     # 최종 background bin MAD
        "bg_bin_MAD_p1":         bg_mad_p1,     # Pass 1 background bin MAD
        # segment 기반 (background-adaptive tier 적용)
        "n_seg_total":           seg_sum["n_seg_total"],
        "n_seg_significant":     seg_sum["n_seg_significant"],
        "n_seg_focal":           seg_sum.get("n_seg_focal", 0),
        "n_seg_standard":        seg_sum.get("n_seg_standard", 0),
        "n_seg_long":            seg_sum.get("n_seg_long", 0),
        "n_seg_weak_gain":       weak_sum["n_seg_weak_gain"],
        "n_seg_weak_loss":       weak_sum["n_seg_weak_loss"],
        "n_diluted_gain":        n_diluted_gain,
        "n_diluted_loss":        n_diluted_loss,
        "n_diluted_conc_gain":   n_dil_conc_gain,
        "n_diluted_conc_loss":   n_dil_conc_loss,
        "diluted_conc_arms":     dil_conc_arms,
        "max_seg_len":           seg_sum["max_seg_len"],
        "max_seg_mean_abs_z":    seg_sum["max_seg_mean_abs_z"],
        "max_seg_peak_abs_z":    seg_sum["max_seg_peak_abs_z"],
        "max_seg_density":       seg_sum["max_seg_density"],
        "total_signif_seg_bins": seg_sum["total_signif_seg_bins"],
        # arm 기반
        "n_positive_arms":       arm_sum["n_positive_arms"],
        "n_arms_strong":         arm_sum.get("n_arms_strong", 0),
        "n_arms_wide":           arm_sum.get("n_arms_wide", 0),
        "max_arm_abs_mean_z":    arm_sum["max_arm_abs_mean_z"],
        "max_arm_frac_outlier":  arm_sum["max_arm_frac_outlier"],
        "arm_positive_list":     arm_sum["arm_positive_list"],
        "sig_seg_arm_list":      sig_seg_arm_list,
        "focal_seg_arm_list":    focal_seg_arm_list,
    })

    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(bam_cols)} done")

df = pd.DataFrame(rows)
df = df.merge(meta, on="AnalysisID", how="left")
out_tsv = OUT_DIR / f"sample_metrics_{DEPTH_LABEL}.tsv"
df.to_csv(out_tsv, sep="\t", index=False)
print(f"\n저장: {out_tsv}  ({len(df)} samples)")

print("\nhealthy train metric 분포 (PoN baseline)")
hsub = df[df["AnalysisID"].isin(train_healthy_aids)]
metric_cols = ["bin_z_MAD","smooth_z_MAD","coherence","n_outlier_smooth",
               "noise_factor","noise_factor_p1","bg_bin_MAD","bg_bin_MAD_p1",
               "n_seg_total","n_seg_significant",
               "n_seg_focal","n_seg_standard","n_seg_long",
               "n_seg_weak_gain","n_seg_weak_loss",
               "n_diluted_gain","n_diluted_loss",
               "n_diluted_conc_gain","n_diluted_conc_loss",
               "noise_factor_capped",
               "max_seg_len","max_seg_mean_abs_z","max_seg_peak_abs_z",
               "max_seg_density","total_signif_seg_bins",
               "n_positive_arms","n_arms_strong","n_arms_wide",
               "max_arm_abs_mean_z","max_arm_frac_outlier"]
if hsub.empty:
    print("  현재 실행 matrix에 healthy Train 샘플이 없어 요약을 생략합니다.")
else:
    for c in metric_cols:
        q = hsub[c].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(3)
        print(f"  {c:>22s}: median={q[0.5]}  Q75={q[0.75]}  Q90={q[0.9]}  Q95={q[0.95]}  Q99={q[0.99]}")
