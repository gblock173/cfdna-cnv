#!/usr/bin/env python3
"""cfDNA diluted-signal helpers for clean, low-amplitude profiles.

Design intent (weak-but-broad):
  Seeds  : any bin/segment with |z| >= 2.0  (not 4.0 — true diluted peaks are weaker)
  Grow   : expand through bins with signed_z >= 1.0 or |z| <= 0.5  (weak-friendly)
  Filter : band must span >= 80 bins (breadth), mean |z| >= 1.0, peak >= 2.0,
           and >= 25% of bins above hi_thr=1.0  (density at the weak-signal level)
  Gate   : sample must be clean (low MAD, low nf) and have at least one peak >= 2.0

All strong-signal thresholds (4.0 / 3.5 / 2.3) have been replaced by
weak-signal equivalents (2.0 / 2.0 / 1.0) so that the grow mechanism
(lo_abs=1.0) and the seed/filter are internally consistent.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

CFDNA_DILUTED_SEG_TIER = {
    "name": "cfdna_diluted",
    "L_min": 80,           # breadth filter — primary FP guard (unchanged)
    "amp_min": 1.0,        # mean |z| of band  (was 2.3 — strong-signal bias)
    "peak_min": 2.0,       # peak |z| of band  (was 4.0 — strong-signal bias)
    "density_min": 0.25,   # fraction of bins >= hi_thr (was 0.15 at hi_thr=3.0)
    "min_peak_bins": 1,    # bins above peak_min (was 2)
}

CFDNA_CLEAN_NF_MAX = 2.5
CFDNA_DILUTED_PEAK_MIN = 2.0    # gate lower bound  (was 4.0)
CFDNA_DILUTED_PEAK_MAX = 12.0   # gate upper bound  (unchanged)
CFDNA_DILUTED_MERGE_GAP_MAX = 30
CFDNA_DILUTED_MERGE_LO_ABS = 1.0       # grow threshold  (unchanged — already weak-friendly)
CFDNA_DILUTED_MERGE_ZERO_ABS = 0.5     # gap tolerance   (unchanged)
CFDNA_DILUTED_SEED_PEAK_MIN = 2.0      # segment seed peak  (was 3.5)
CFDNA_DILUTED_BIN_SEED_MIN = 2.0       # bin seed threshold  (was 4.0)
CFDNA_DILUTED_SEED_NMS_BIN = 15        # NMS window  (unchanged)
CFDNA_DILUTED_HI_THR = 1.0             # density base  (was 3.0 — recalibrated to weak signal)


def diluted_params_from_config(cfg: Optional[dict] = None) -> dict:
    c = (cfg or {}).get("caller", {})
    return {
        "tier": {
            "name": "cfdna_diluted",
            "L_min": int(c.get("cfdna_diluted_L_min", CFDNA_DILUTED_SEG_TIER["L_min"])),
            "amp_min": float(c.get("cfdna_diluted_amp_min", CFDNA_DILUTED_SEG_TIER["amp_min"])),
            "peak_min": float(c.get("cfdna_diluted_peak_min", CFDNA_DILUTED_SEG_TIER["peak_min"])),
            "density_min": float(c.get("cfdna_diluted_density_min",
                                       CFDNA_DILUTED_SEG_TIER["density_min"])),
            "min_peak_bins": int(c.get("cfdna_diluted_min_peak_bins",
                                      CFDNA_DILUTED_SEG_TIER["min_peak_bins"])),
        },
        "merge_gap_max": int(c.get("cfdna_diluted_merge_gap_max", CFDNA_DILUTED_MERGE_GAP_MAX)),
        "merge_lo_abs": float(c.get("cfdna_diluted_merge_lo_abs", CFDNA_DILUTED_MERGE_LO_ABS)),
        "merge_zero_abs": float(c.get("cfdna_diluted_merge_zero_abs", CFDNA_DILUTED_MERGE_ZERO_ABS)),
        "seed_peak_min": float(c.get("cfdna_diluted_seed_peak_min", CFDNA_DILUTED_SEED_PEAK_MIN)),
        "bin_seed_min": float(c.get("cfdna_diluted_bin_seed_min", CFDNA_DILUTED_BIN_SEED_MIN)),
        "seed_nms_bin": int(c.get("cfdna_diluted_seed_nms_bin", CFDNA_DILUTED_SEED_NMS_BIN)),
        # Density-measurement threshold for diluted bands. Historically coupled to
        # z_call_thr (segment hi threshold); when z_call_thr is raised for the
        # segment caller this also tightens diluted density, which over-suppresses
        # weak low-depth bands. Allow an explicit override decoupled from z_call_thr.
        # Default preserves the legacy behaviour (= z_call_thr).
        "hi_thr": float(c.get("cfdna_diluted_hi_thr",
                              c.get("z_call_thr", CFDNA_DILUTED_HI_THR))),
        # Gate thresholds (is_cfdna_diluted_profile) — exposed for depth-specific
        # tuning. Defaults preserve the 100G baseline. gate_peak_min falls back to
        # the shared cfdna_diluted_peak_min, then to the gate constant.
        "gate_peak_min": float(c.get("cfdna_diluted_gate_peak_min",
                                     c.get("cfdna_diluted_peak_min", CFDNA_DILUTED_PEAK_MIN))),
        "gate_nf_max": float(c.get("cfdna_diluted_nf_max", CFDNA_CLEAN_NF_MAX)),
        "gate_noisy_bin_mad": float(c.get("noisy_bin_mad", 1.81)),
    }


def is_cfdna_diluted_profile(bin_z_mad: float,
                             noise_factor: float,
                             max_peak_abs_z: float,
                             noisy_bin_mad: float = 1.81,
                             nf_max: float = CFDNA_CLEAN_NF_MAX,
                             peak_min: float = CFDNA_DILUTED_PEAK_MIN,
                             peak_max: float = CFDNA_DILUTED_PEAK_MAX) -> bool:
    """True for clean cfDNA with moderate peak amplitude (diluted CNV, not noise/tissue)."""
    if not (bin_z_mad == bin_z_mad and noise_factor == noise_factor):
        return False
    if bin_z_mad >= noisy_bin_mad:
        return False
    if noise_factor >= nf_max:
        return False
    if max_peak_abs_z < peak_min or max_peak_abs_z > peak_max:
        return False
    return True


def seg_passes_cfdna_diluted(seg: dict, tier: Optional[dict] = None,
                             z_smooth: Optional[np.ndarray] = None) -> bool:
    """Low-amplitude merged band tier for diluted cfDNA (longer than standard segment)."""
    t = tier or CFDNA_DILUTED_SEG_TIER
    if seg.get("from_subseg"):
        return False
    if seg["length"] < t["L_min"]:
        return False
    if "L_max" in t and seg["length"] > t["L_max"]:
        return False
    if not (seg["mean_abs_z"] >= t["amp_min"]
            and seg["peak_abs_z"] >= t["peak_min"]
            and seg["density"] >= t["density_min"]):
        return False
    min_peak_bins = t.get("min_peak_bins", 0)
    if min_peak_bins > 0 and z_smooth is not None:
        sub = z_smooth[seg["start"]:seg["end"] + 1]
        fin = np.isfinite(sub)
        signed = seg["direction"] * sub[fin]
        if int((signed >= t["peak_min"]).sum()) < min_peak_bins:
            return False
    return True


def compute_seg_stats(z_smooth: np.ndarray,
                      start: int,
                      end: int,
                      direction: int,
                      hi_thr: float = CFDNA_DILUTED_HI_THR) -> Optional[dict]:
    sub = z_smooth[start:end + 1]
    fin = np.isfinite(sub)
    if not fin.any():
        return None
    signed = direction * sub
    abs_v = np.abs(sub[fin])
    length = end - start + 1
    n_strict = int((signed[fin] >= hi_thr).sum())
    fin_idx = np.where(fin)[0]
    peak_local = int(fin_idx[int(np.nanargmax(abs_v))])
    return {
        "start": int(start),
        "end": int(end),
        "direction": int(direction),
        "length": int(length),
        "mean_abs_z": float(abs_v.mean()),
        "peak_abs_z": float(abs_v.max()),
        "n_strict": n_strict,
        "density": float(n_strict / length) if length > 0 else 0.0,
        "peak_bin": int(start + peak_local),
    }


def gap_bridge_ok(z_smooth: np.ndarray,
                  gap_start: int,
                  gap_end: int,
                  direction: int,
                  lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                  zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS,
                  min_ok_frac: float = 0.35) -> bool:
    """Allow merging across dips: same-sign above lo_abs, or near-zero trough bins."""
    if gap_start > gap_end:
        return True
    gap = z_smooth[gap_start:gap_end + 1]
    fin = np.isfinite(gap)
    if not fin.any():
        return True
    signed = direction * gap[fin]
    ok = (signed >= lo_abs) | (np.abs(gap[fin]) <= zero_abs)
    return float(ok.sum()) / float(len(signed)) >= min_ok_frac


def _bin_ok_for_grow(z_smooth: np.ndarray,
                     idx: int,
                     direction: int,
                     lo_abs: float,
                     zero_abs: float,
                     bin_chrom: Optional[np.ndarray],
                     chrom) -> bool:
    if idx < 0 or idx >= len(z_smooth):
        return False
    zv = z_smooth[idx]
    if not np.isfinite(zv):
        return False
    if bin_chrom is not None and bin_chrom[idx] != chrom:
        return False
    signed = direction * zv
    return signed >= lo_abs or abs(zv) <= zero_abs


def grow_run_from_seed(z_smooth: np.ndarray,
                       seed_idx: int,
                       direction: int,
                       lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                       zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS,
                       bin_chrom: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """Expand from peak seed through same-sign bins; tolerate near-zero dips."""
    chrom = bin_chrom[seed_idx] if bin_chrom is not None else None
    left = seed_idx
    while _bin_ok_for_grow(z_smooth, left - 1, direction, lo_abs, zero_abs, bin_chrom, chrom):
        left -= 1
    right = seed_idx
    while _bin_ok_for_grow(z_smooth, right + 1, direction, lo_abs, zero_abs, bin_chrom, chrom):
        right += 1
    return left, right


def collect_diluted_seeds(z_smooth: np.ndarray,
                          segments: Optional[Sequence[dict]] = None,
                          bin_seed_min: float = CFDNA_DILUTED_BIN_SEED_MIN,
                          seed_peak_min: float = CFDNA_DILUTED_SEED_PEAK_MIN,
                          seed_nms_bin: int = CFDNA_DILUTED_SEED_NMS_BIN) -> List[Tuple[int, int]]:
    """Peak-bin seeds plus hysteresis segment peaks; NMS within seed_nms_bin."""
    seeds: List[Tuple[int, int]] = []
    fin = np.isfinite(z_smooth)
    abs_z = np.where(fin, np.abs(z_smooth), 0.0)
    for i in np.where(fin & (abs_z >= bin_seed_min))[0]:
        seeds.append((int(i), 1 if z_smooth[i] > 0 else -1))
    if segments:
        for seg in segments:
            if seg.get("from_subseg") or seg.get("from_diluted_merge"):
                continue
            peak_i = int(seg.get("peak_bin", (seg["start"] + seg["end"]) // 2))
            if 0 <= peak_i < len(z_smooth) and fin[peak_i] and abs_z[peak_i] >= seed_peak_min:
                seeds.append((peak_i, int(seg["direction"])))
    if not seeds:
        return []
    seeds.sort(key=lambda t: abs_z[t[0]], reverse=True)
    kept: List[Tuple[int, int]] = []
    for idx, direction in seeds:
        if any(abs(idx - k) <= seed_nms_bin and direction == kd for k, kd in kept):
            continue
        kept.append((idx, direction))
    return kept


def extract_diluted_runs(z_smooth: np.ndarray,
                         segments: Optional[Sequence[dict]] = None,
                         bin_chrom: Optional[np.ndarray] = None,
                         lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                         zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS,
                         bin_seed_min: float = CFDNA_DILUTED_BIN_SEED_MIN,
                         seed_peak_min: float = CFDNA_DILUTED_SEED_PEAK_MIN,
                         seed_nms_bin: int = CFDNA_DILUTED_SEED_NMS_BIN) -> List[Tuple[int, int, int]]:
    """Return list of (start, end, direction) runs from peak grow."""
    seeds = collect_diluted_seeds(
        z_smooth, segments=segments,
        bin_seed_min=bin_seed_min, seed_peak_min=seed_peak_min,
        seed_nms_bin=seed_nms_bin)
    runs: List[Tuple[int, int, int]] = []
    for seed_idx, direction in seeds:
        left, right = grow_run_from_seed(
            z_smooth, seed_idx, direction, lo_abs=lo_abs, zero_abs=zero_abs,
            bin_chrom=bin_chrom)
        runs.append((left, right, direction))
    return runs


def _merge_touching_runs(runs: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    if not runs:
        return []
    out: List[Tuple[int, int, int]] = []
    for direction in (1, -1):
        group = sorted([r for r in runs if r[2] == direction], key=lambda r: r[0])
        for start, end, _ in group:
            if out and out[-1][2] == direction and start <= out[-1][1] + 1:
                out[-1] = (out[-1][0], max(out[-1][1], end), direction)
            else:
                out.append((start, end, direction))
    return out


def cluster_diluted_runs(runs: List[Tuple[int, int, int]],
                         z_smooth: np.ndarray,
                         gap_max: int = CFDNA_DILUTED_MERGE_GAP_MAX,
                         lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                         zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS) -> List[Tuple[int, int, int]]:
    """Cluster same-direction runs; bridge gaps with dip tolerance."""
    runs = _merge_touching_runs(runs)
    clustered: List[Tuple[int, int, int]] = []
    for direction in (1, -1):
        group = sorted([r for r in runs if r[2] == direction], key=lambda r: r[0])
        i = 0
        while i < len(group):
            cs, ce, _ = group[i]
            j = i + 1
            while j < len(group):
                ns, ne, _ = group[j]
                if ns > ce + gap_max + 1:
                    break
                if ns <= ce + 1 or gap_bridge_ok(z_smooth, ce + 1, ns - 1, direction, lo_abs, zero_abs):
                    ce = max(ce, ne)
                    j += 1
                else:
                    break
            clustered.append((cs, ce, direction))
            i = j if j > i + 1 else i + 1
    return clustered


def build_diluted_band_segments(z_smooth: np.ndarray,
                                segments: Optional[Sequence[dict]] = None,
                                bin_chrom: Optional[np.ndarray] = None,
                                tier: Optional[dict] = None,
                                gap_max: int = CFDNA_DILUTED_MERGE_GAP_MAX,
                                lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                                zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS,
                                hi_thr: float = CFDNA_DILUTED_HI_THR,
                                bin_seed_min: float = CFDNA_DILUTED_BIN_SEED_MIN,
                                seed_peak_min: float = CFDNA_DILUTED_SEED_PEAK_MIN,
                                seed_nms_bin: int = CFDNA_DILUTED_SEED_NMS_BIN) -> List[dict]:
    """v2: peak seed → grow → cluster → long-span tier filter."""
    t = tier or CFDNA_DILUTED_SEG_TIER
    runs = extract_diluted_runs(
        z_smooth, segments=segments, bin_chrom=bin_chrom,
        lo_abs=lo_abs, zero_abs=zero_abs,
        bin_seed_min=bin_seed_min, seed_peak_min=seed_peak_min,
        seed_nms_bin=seed_nms_bin)
    clusters = cluster_diluted_runs(runs, z_smooth, gap_max=gap_max, lo_abs=lo_abs, zero_abs=zero_abs)
    merged_out: List[dict] = []
    for start, end, direction in clusters:
        stats = compute_seg_stats(z_smooth, start, end, direction, hi_thr=hi_thr)
        if stats is not None and seg_passes_cfdna_diluted(stats, t, z_smooth=z_smooth):
            stats["from_diluted_merge"] = True
            merged_out.append(stats)
    return merged_out


def merge_diluted_band_segments(segments: Sequence[dict],
                                z_smooth: np.ndarray,
                                tier: Optional[dict] = None,
                                gap_max: int = CFDNA_DILUTED_MERGE_GAP_MAX,
                                lo_abs: float = CFDNA_DILUTED_MERGE_LO_ABS,
                                zero_abs: float = CFDNA_DILUTED_MERGE_ZERO_ABS,
                                hi_thr: float = CFDNA_DILUTED_HI_THR,
                                seed_peak_min: float = CFDNA_DILUTED_SEED_PEAK_MIN,
                                bin_chrom: Optional[np.ndarray] = None,
                                bin_seed_min: float = CFDNA_DILUTED_BIN_SEED_MIN,
                                seed_nms_bin: int = CFDNA_DILUTED_SEED_NMS_BIN) -> List[dict]:
    """Backward-compatible alias for v2 diluted band builder."""
    return build_diluted_band_segments(
        z_smooth, segments=segments, bin_chrom=bin_chrom, tier=tier,
        gap_max=gap_max, lo_abs=lo_abs, zero_abs=zero_abs, hi_thr=hi_thr,
        bin_seed_min=bin_seed_min, seed_peak_min=seed_peak_min,
        seed_nms_bin=seed_nms_bin)


def cfdna_diluted_band_segments(segments: Sequence[dict],
                                z_smooth: np.ndarray,
                                bin_z_mad: float,
                                noise_factor: float,
                                max_peak_abs_z: float,
                                noisy_bin_mad: float = 1.81,
                                bin_chrom: Optional[np.ndarray] = None,
                                cfg: Optional[dict] = None) -> List[dict]:
    p = diluted_params_from_config(cfg)
    # Gate params come from config (with the explicit noisy_bin_mad arg taking
    # precedence to preserve existing call sites that pass it directly).
    nbm = noisy_bin_mad if noisy_bin_mad != 1.81 else p["gate_noisy_bin_mad"]
    if not is_cfdna_diluted_profile(bin_z_mad, noise_factor, max_peak_abs_z,
                                    noisy_bin_mad=nbm,
                                    nf_max=p["gate_nf_max"],
                                    peak_min=p["gate_peak_min"]):
        return []
    return build_diluted_band_segments(
        z_smooth, segments=segments, bin_chrom=bin_chrom,
        tier=p["tier"],
        gap_max=p["merge_gap_max"],
        lo_abs=p["merge_lo_abs"],
        zero_abs=p["merge_zero_abs"],
        hi_thr=p["hi_thr"],
        bin_seed_min=p["bin_seed_min"],
        seed_peak_min=p["seed_peak_min"],
        seed_nms_bin=p["seed_nms_bin"],
    )


def count_cfdna_diluted_bands(segments: Sequence[dict],
                              z_smooth: np.ndarray,
                              bin_z_mad: float,
                              noise_factor: float,
                              max_peak_abs_z: float,
                              noisy_bin_mad: float = 1.81,
                              bin_chrom: Optional[np.ndarray] = None,
                              cfg: Optional[dict] = None) -> int:
    return len(cfdna_diluted_band_segments(
        segments, z_smooth, bin_z_mad, noise_factor, max_peak_abs_z,
        noisy_bin_mad=noisy_bin_mad, bin_chrom=bin_chrom, cfg=cfg))
