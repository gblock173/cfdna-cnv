#!/usr/bin/env python3
"""Arm-level statistics helpers (import-safe, no batch side effects)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

ARM_TIERS: List[dict] = [
    {"name": "strong", "mean_min": 2.5, "frac_min": 0.30},
    {"name": "wide",   "mean_min": 1.2, "frac_min": 0.50},
]

CFDNA_ARM_TIERS: List[dict] = [
    {"name": "strong", "mean_min": 1.2, "frac_min": 0.15},
    {"name": "wide",   "mean_min": 0.9, "frac_min": 0.25},
]

# Joint arm-mean and outlier-density criteria for diluted-gain concordance.
CFDNA_DILUTED_CONC_TIERS: List[dict] = [
    {"name": "conc", "mean_min": 1.8, "frac_min": 0.25, "frac_z3_min": 0.15},
]

Z_CALL_THR = 3.0
MIN_BINS_PER_ARM = 30
# z threshold for the strong-outlier-density (frac_z3) concordance check.
CFDNA_CONC_Z3_THR = 3.0


def build_bin_to_arm(chrs_sorted, starts_sorted, arm_coords, autosome_chroms):
    autosome_set = set(autosome_chroms)
    arm_idx_of_arm = {
        ac.arm: i for i, ac in enumerate(arm_coords) if ac.chrom in autosome_set
    }
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


def compute_arm_stats(z_smooth, bin_to_arm, n_arms,
                      z_call_thr: float = Z_CALL_THR,
                      min_bins_per_arm: int = MIN_BINS_PER_ARM) -> dict:
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
        out["mean_z"][a] = float(np.mean(z))
        out["frac_pos"][a] = float(np.mean(z > z_call_thr))
        out["frac_neg"][a] = float(np.mean(z < -z_call_thr))
    return out


def summarize_arms(arm_stats, arm_names, tiers: Sequence[dict] = ARM_TIERS) -> dict:
    mean_z = arm_stats["mean_z"]
    frac_pos = arm_stats["frac_pos"]
    frac_neg = arm_stats["frac_neg"]
    finite = np.isfinite(mean_z)
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
    abs_mean = np.where(finite, np.abs(mean_z), 0.0)
    frac_any = np.where(
        finite,
        np.maximum(np.nan_to_num(frac_pos), np.nan_to_num(frac_neg)),
        0.0,
    )
    any_mask = np.zeros(len(mean_z), dtype=bool)
    for t in tiers:
        m = finite & (abs_mean >= t["mean_min"]) & (frac_any >= t["frac_min"])
        out[f"n_arms_{t['name']}"] = int(m.sum())
        any_mask |= m
    pos_arms = []
    for i in np.where(any_mask)[0]:
        sign = "+" if mean_z[i] > 0 else "-"
        pos_arms.append(f"{arm_names[i]}{sign}")
    out["n_positive_arms"] = int(any_mask.sum())
    out["max_arm_abs_mean_z"] = float(abs_mean.max())
    out["max_arm_frac_outlier"] = float(frac_any.max())
    out["arm_positive_list"] = ",".join(pos_arms)
    return out


def arm_signed_name(arm_names: Sequence[str], arm_idx: int, mean_z: float) -> str:
    sign = "+" if mean_z > 0 else "-"
    return f"{arm_names[arm_idx]}{sign}"


def diluted_bands_arm_concordant(
        diluted_segs: Sequence[dict],
        bin_arm,
        cfdna_arm_stats: dict,
        arm_names: Sequence[str],
        tiers: Sequence[dict] = CFDNA_DILUTED_CONC_TIERS,
        min_overlap_bins: int = 40,
        gain_only: bool = True,
):
    """Filter diluted bands to those corroborated by a concordant arm-mean shift.

    A diluted band (breadth axis) is kept only if it overlaps (>= min_overlap_bins)
    an arm whose cfDNA arm statistics (amplitude axis) shift in the SAME direction
    and pass one of the concordance tiers. Requiring multiple semi-independent
    measurements to agree suppresses healthy noise (which rarely produces a long
    coherent run AND a concordant arm-mean shift AND a high strong-outlier density)
    while keeping true weak-but-broad tumor signal (which produces all three).

    Concordance tier conditions (all must hold for the overlapping arm):
        abs(mean_z)   >= mean_min     (arm-mean amplitude, computed on z_smooth)
        frac_any      >= frac_min     (fraction with |z| > weak z, e.g. 1.0)
        frac_any_z3   >= frac_z3_min  (fraction with |z| > 3 — strong-outlier density)

    gain_only : if True (default) only GAIN-direction bands are evaluated. The LOSS
                channel was found to add 0 true cancer detections while contributing
                healthy FPs, so it is disabled by default.

    cfdna_arm_stats : output of compute_arm_stats(...). Must contain frac_pos/frac_neg
                      at the weak z (e.g. 1.0); for the frac_z3 check it must also
                      contain frac_pos_z3/frac_neg_z3 (computed at z=3.0). If the z3
                      keys are absent the frac_z3_min condition is skipped (back-compat).

    Returns (n_concordant_gain, n_concordant_loss, concordant_arm_list_str).
    """
    name_to_idx = {n: i for i, n in enumerate(arm_names)}
    mean_z   = cfdna_arm_stats["mean_z"]
    frac_pos = cfdna_arm_stats["frac_pos"]
    frac_neg = cfdna_arm_stats["frac_neg"]
    frac_pos_z3 = cfdna_arm_stats.get("frac_pos_z3")
    frac_neg_z3 = cfdna_arm_stats.get("frac_neg_z3")
    has_z3 = frac_pos_z3 is not None and frac_neg_z3 is not None

    n_gain = n_loss = 0
    arms_out: List[str] = []
    for seg in diluted_segs:
        direction = int(seg["direction"])
        if gain_only and direction <= 0:
            continue
        sub = bin_arm[seg["start"]:seg["end"] + 1]
        names = [a for a in sub if a is not None]
        if not names:
            continue
        uniq, counts = np.unique(np.array(names), return_counts=True)
        concordant_arm = None
        for nm, cnt in zip(uniq, counts):
            if cnt < min_overlap_bins or nm not in name_to_idx:
                continue
            ai = name_to_idx[nm]
            mz = mean_z[ai]
            if not np.isfinite(mz) or (1 if mz > 0 else -1) != direction:
                continue
            abs_mean = abs(float(mz))
            frac_any = float(max(np.nan_to_num(frac_pos[ai]),
                                 np.nan_to_num(frac_neg[ai])))
            frac_z3 = float(max(np.nan_to_num(frac_pos_z3[ai]),
                                np.nan_to_num(frac_neg_z3[ai]))) if has_z3 else None
            for t in tiers:
                if abs_mean < t["mean_min"] or frac_any < t["frac_min"]:
                    continue
                z3_min = t.get("frac_z3_min")
                if z3_min is not None and has_z3 and frac_z3 < z3_min:
                    continue
                concordant_arm = f"{nm}{'+' if direction > 0 else '-'}"
                break
            if concordant_arm:
                break
        if concordant_arm:
            arms_out.append(concordant_arm)
            if direction > 0:
                n_gain += 1
            else:
                n_loss += 1
    return n_gain, n_loss, ",".join(sorted(set(arms_out)))


def tissue_concordant_arms(
        cfdna_arm_stats: dict,
        arm_names: Sequence[str],
        tissue_arm_list: Sequence[str],
        tiers: Sequence[dict] = CFDNA_ARM_TIERS,
) -> List[str]:
    """Return tissue arm names concordant in cfDNA (same direction + cfDNA tier pass)."""
    name_to_idx = {n: i for i, n in enumerate(arm_names)}
    out: List[str] = []
    for ta in tissue_arm_list:
        ta = ta.strip()
        if not ta or ta[-1] not in "+-":
            continue
        base = ta[:-1]
        want_sign = 1 if ta[-1] == "+" else -1
        if base not in name_to_idx:
            continue
        i = name_to_idx[base]
        mz = cfdna_arm_stats["mean_z"][i]
        if not np.isfinite(mz) or np.sign(mz) != want_sign:
            continue
        abs_mean = abs(float(mz))
        frac_any = float(max(cfdna_arm_stats["frac_pos"][i], cfdna_arm_stats["frac_neg"][i]))
        for t in tiers:
            if abs_mean >= t["mean_min"] and frac_any >= t["frac_min"]:
                out.append(ta)
                break
    return out
