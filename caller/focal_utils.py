#!/usr/bin/env python3
"""Shared focal CNV peak refinement helpers.

The caller's base hysteresis segments are intentionally permissive so they do
not fragment broad events. Focal calls need the opposite behavior: keep the
peak-centered event and avoid letting broad shoulders define the focal bounds.

SNR role separation for refined subsegments:
  - Classic : mean + peak SNR vs global bg_mad
  - Spike (global) : peak SNR vs global bg_mad
  - Spike (local)  : peak_delta / quiet_flank_mad AND peak_delta / bg_mad
                     (quiet side only — opposite CNV at adjacent flank excluded)
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _focal_classic_passes(seg: dict, focal_tier: dict,
                          bg_mad: Optional[float] = None) -> bool:
    """Classic focal: mean + peak SNR both required on trimmed window."""
    if seg["length"] < focal_tier["L_min"]:
        return False
    if "L_max" in focal_tier and seg["length"] > focal_tier["L_max"]:
        return False
    if not (seg["mean_abs_z"] >= focal_tier["amp_min"]
            and seg["peak_abs_z"] >= focal_tier["peak_min"]
            and seg["density"] >= focal_tier["density_min"]):
        return False
    if bg_mad is not None and bg_mad > 0:
        snr_min = focal_tier.get("snr_min")
        if snr_min is not None and seg["mean_abs_z"] / bg_mad < snr_min:
            return False
        peak_snr_min = focal_tier.get("peak_snr_min")
        if peak_snr_min is not None and seg["peak_abs_z"] / bg_mad < peak_snr_min:
            return False
    return True


def _focal_spike_local_passes(seg: dict,
                              bg_mad: Optional[float] = None,
                              spike_base_max: float = 3.0,
                              spike_peak_delta_min: float = 5.0,
                              local_spike_snr_min: float = 3.0,
                              spike_delta_snr_min: float = 4.0,
                              noise_ref_mad: float = 1.0) -> bool:
    """Local spike SNR: quiet-flank MAD + incremental delta SNR vs bg_mad."""
    local_base = seg.get("local_baseline")
    quiet_mad = seg.get("local_quiet_flank_mad", seg.get("local_flank_mad"))
    if local_base is None or quiet_mad is None:
        return False
    if not (np.isfinite(local_base) and np.isfinite(quiet_mad)):
        return False
    local_base = float(local_base)
    quiet_mad = float(quiet_mad)
    peak_delta = seg["peak_abs_z"] - local_base
    if local_base > spike_base_max:
        return False
    if peak_delta < spike_peak_delta_min:
        return False
    local_denom = max(quiet_mad, float(noise_ref_mad))
    if (peak_delta / local_denom) < local_spike_snr_min:
        return False
    if bg_mad is not None and bg_mad > 0:
        if (peak_delta / bg_mad) < spike_delta_snr_min:
            return False
    return True


def _focal_spike_passes(seg: dict, focal_tier: dict,
                        bg_mad: Optional[float] = None,
                        peak_snr_min: float = 6.0,
                        spike_max_len: int = 15,
                        spike_base_max: float = 3.0,
                        spike_peak_delta_min: float = 5.0,
                        local_spike_snr_min: float = 3.0,
                        spike_delta_snr_min: float = 4.0,
                        noise_ref_mad: float = 1.0) -> bool:
    """Spike focal: global peak SNR or local quiet-flank spike SNR."""
    if seg["length"] < focal_tier["L_min"]:
        return False
    if "L_max" in focal_tier and seg["length"] > focal_tier["L_max"]:
        return False
    if seg["peak_abs_z"] < focal_tier["peak_min"]:
        return False
    if seg["density"] < focal_tier["density_min"]:
        return False

    global_ok = False
    if bg_mad is not None and bg_mad > 0:
        global_ok = seg["peak_abs_z"] / bg_mad >= peak_snr_min
    local_ok = _focal_spike_local_passes(
        seg,
        bg_mad=bg_mad,
        spike_base_max=spike_base_max,
        spike_peak_delta_min=spike_peak_delta_min,
        local_spike_snr_min=local_spike_snr_min,
        spike_delta_snr_min=spike_delta_snr_min,
        noise_ref_mad=noise_ref_mad,
    )
    if not (global_ok or local_ok):
        return False

    if seg["length"] <= spike_max_len:
        return True
    if seg["density"] >= 0.70 and seg["peak_abs_z"] >= focal_tier["amp_min"]:
        return True
    return False


def focal_candidate_passes(seg: dict, focal_tier: dict,
                           bg_mad: Optional[float] = None,
                           spike_base_max: float = 3.0,
                           spike_peak_delta_min: float = 5.0,
                           local_spike_snr_min: float = 3.0,
                           spike_delta_snr_min: float = 4.0,
                           noise_ref_mad: float = 1.0) -> bool:
    """Classic or spike path (global or local quiet-flank SNR)."""
    return (_focal_classic_passes(seg, focal_tier, bg_mad=bg_mad)
            or _focal_spike_passes(
                seg, focal_tier, bg_mad=bg_mad,
                spike_base_max=spike_base_max,
                spike_peak_delta_min=spike_peak_delta_min,
                local_spike_snr_min=local_spike_snr_min,
                spike_delta_snr_min=spike_delta_snr_min,
                noise_ref_mad=noise_ref_mad))


def focal_boundary_abs(peak_abs: float, bg_mad: Optional[float],
                       hi_abs: float = 2.0,
                       boundary_min_abs: float = 3.0,
                       peak_frac: float = 0.45,
                       bg_mad_mult: float = 2.0) -> float:
    """Return the directional z threshold used to trim a focal peak."""
    vals = [float(hi_abs), float(boundary_min_abs), float(peak_abs) * float(peak_frac)]
    if bg_mad is not None and np.isfinite(bg_mad) and bg_mad > 0:
        vals.append(float(bg_mad) * float(bg_mad_mult))
    return float(max(vals))


def _directional_flank_mad(scores: np.ndarray) -> Optional[float]:
    if scores.size == 0:
        return None
    med = float(np.nanmedian(scores))
    mad = float(np.nanmedian(np.abs(scores - med)))
    if not np.isfinite(mad) or mad < 0.05:
        mad = 0.05
    return mad


def _local_flank_baseline(z_smooth: np.ndarray,
                          bin_chrom: Optional[Sequence],
                          start: int,
                          end: int,
                          direction: int,
                          flank_bins: int,
                          min_flank_bins: int) -> Tuple[
                              Optional[float], int, Optional[float], Optional[float]]:
    """Directional local baseline + pooled/quiet flank MAD around a candidate."""
    n = len(z_smooth)
    chrom = bin_chrom[start] if bin_chrom is not None else None
    side_scores: List[np.ndarray] = []
    min_side_bins = max(5, min_flank_bins // 2)
    for a, b in ((max(0, start - flank_bins), start),
                 (end + 1, min(n, end + 1 + flank_bins))):
        if a >= b:
            continue
        vals = z_smooth[a:b]
        if bin_chrom is not None:
            chroms = np.asarray(bin_chrom[a:b])
            vals = vals[chroms == chrom]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            side_scores.append(direction * vals)
    if not side_scores:
        return None, 0, None, None
    merged = np.concatenate(side_scores)
    if len(merged) < min_flank_bins:
        return None, int(len(merged)), None, None
    local_base = float(max(0.0, np.nanmedian(merged)))
    flank_mad = _directional_flank_mad(merged)
    side_mads = [
        _directional_flank_mad(s) for s in side_scores if len(s) >= min_side_bins
    ]
    side_mads = [m for m in side_mads if m is not None]
    quiet_flank_mad = min(side_mads) if side_mads else flank_mad
    return local_base, int(len(merged)), flank_mad, quiet_flank_mad


def _local_excess_passes(peak_a: float,
                         mean_a: float,
                         local_base: Optional[float],
                         local_peak_delta_min: float = 3.0,
                         local_mean_delta_min: float = 1.5,
                         local_excess_mode: str = "simple",
                         spike_base_max: float = 3.0,
                         spike_peak_delta_min: float = 5.0,
                         embedded_base_min: float = 15.0,
                         embedded_peak_delta_min: float = 4.0,
                         embedded_peak_delta_max: float = 5.5,
                         embedded_mean_delta_min: float = 3.0) -> bool:
    """4차 local excess: flank baseline 대비 peak/mean elevation."""
    if local_base is None:
        return True
    peak_delta = peak_a - local_base
    mean_delta = mean_a - local_base
    if local_excess_mode == "dual":
        if local_base <= spike_base_max and peak_delta >= spike_peak_delta_min:
            return True
        if (local_base >= embedded_base_min
                and embedded_peak_delta_min <= peak_delta <= embedded_peak_delta_max
                and mean_delta >= embedded_mean_delta_min):
            return True
        return False
    return (peak_delta >= local_peak_delta_min
            and mean_delta >= local_mean_delta_min)


def extract_focal_subsegments_refined(
    segments: Sequence[dict],
    z_smooth: np.ndarray,
    bin_chrom: Optional[Sequence],
    focal_tier: dict,
    bg_mad: Optional[float],
    hi_abs: float = 2.0,
    boundary_min_abs: float = 3.0,
    boundary_peak_frac: float = 0.45,
    boundary_bg_mad_mult: float = 2.0,
    flank_bins: int = 50,
    min_flank_bins: int = 10,
    local_peak_delta_min: float = 3.0,
    local_mean_delta_min: float = 1.5,
    local_excess_mode: str = "simple",
    spike_base_max: float = 3.0,
    spike_peak_delta_min: float = 5.0,
    local_spike_snr_min: float = 3.0,
    spike_delta_snr_min: float = 4.0,
    noise_ref_mad: float = 1.0,
    embedded_base_min: float = 15.0,
    embedded_peak_delta_min: float = 4.0,
    embedded_peak_delta_max: float = 5.5,
    embedded_mean_delta_min: float = 3.0,
) -> List[dict]:
    """Extract one peak-centered focal event from each base segment."""
    L_min = focal_tier["L_min"]
    L_max = focal_tier["L_max"]
    out: List[dict] = []
    for s in segments:
        sub = z_smooth[s["start"]:s["end"] + 1]
        if not np.isfinite(sub).any():
            continue
        chrom_sub = bin_chrom[s["start"]:s["end"] + 1] if bin_chrom is not None else None
        abs_sub = np.abs(sub)
        peak_local = int(np.nanargmax(abs_sub))
        peak_val = float(abs_sub[peak_local])
        if peak_val < focal_tier["peak_min"]:
            continue
        direction = 1 if sub[peak_local] > 0 else -1
        peak_chrom = chrom_sub[peak_local] if chrom_sub is not None else None
        boundary_abs = focal_boundary_abs(
            peak_val,
            bg_mad,
            hi_abs=hi_abs,
            boundary_min_abs=boundary_min_abs,
            peak_frac=boundary_peak_frac,
            bg_mad_mult=boundary_bg_mad_mult,
        )

        l = peak_local
        while l > 0:
            cand = sub[l - 1]
            if not np.isfinite(cand):
                break
            if chrom_sub is not None and chrom_sub[l - 1] != peak_chrom:
                break
            if direction * cand < boundary_abs:
                break
            if (peak_local - (l - 1) + 1) > L_max:
                break
            l -= 1

        r = peak_local
        while r < len(sub) - 1:
            cand = sub[r + 1]
            if not np.isfinite(cand):
                break
            if chrom_sub is not None and chrom_sub[r + 1] != peak_chrom:
                break
            if direction * cand < boundary_abs:
                break
            if ((r + 1) - l + 1) > L_max:
                break
            r += 1

        win = sub[l:r + 1]
        win_f = win[np.isfinite(win)]
        win_len = r - l + 1
        if win_len < L_min:
            continue
        abs_w = np.abs(win_f)
        mean_a = float(abs_w.mean())
        peak_a = float(abs_w.max())
        n_strict = int((direction * win_f >= boundary_abs).sum())
        density = n_strict / win_len if win_len > 0 else 0.0

        cand_start = int(s["start"] + l)
        cand_end = int(s["start"] + r)
        local_base, local_n, flank_mad, quiet_flank_mad = _local_flank_baseline(
            z_smooth,
            bin_chrom,
            cand_start,
            cand_end,
            direction,
            flank_bins,
            min_flank_bins,
        )
        if local_base is not None:
            if not _local_excess_passes(
                    peak_a, mean_a, local_base,
                    local_peak_delta_min=local_peak_delta_min,
                    local_mean_delta_min=local_mean_delta_min,
                    local_excess_mode=local_excess_mode,
                    spike_base_max=spike_base_max,
                    spike_peak_delta_min=spike_peak_delta_min,
                    embedded_base_min=embedded_base_min,
                    embedded_peak_delta_min=embedded_peak_delta_min,
                    embedded_peak_delta_max=embedded_peak_delta_max,
                    embedded_mean_delta_min=embedded_mean_delta_min):
                continue

        cand_seg = {
            "start": cand_start,
            "end": cand_end,
            "peak_bin": int(s["start"] + peak_local),
            "direction": int(direction),
            "length": int(win_len),
            "mean_abs_z": mean_a,
            "peak_abs_z": peak_a,
            "n_strict": n_strict,
            "density": float(density),
            "from_subseg": True,
            "boundary_abs": float(boundary_abs),
            "local_baseline": float(local_base) if local_base is not None else np.nan,
            "local_flank_mad": float(flank_mad) if flank_mad is not None else np.nan,
            "local_quiet_flank_mad": (
                float(quiet_flank_mad) if quiet_flank_mad is not None else np.nan),
            "local_flank_bins": int(local_n),
            "parent_start": int(s["start"]),
            "parent_end": int(s["end"]),
            "parent_length": int(s["length"]),
        }
        if focal_candidate_passes(
                cand_seg, focal_tier, bg_mad=bg_mad,
                spike_base_max=spike_base_max,
                spike_peak_delta_min=spike_peak_delta_min,
                local_spike_snr_min=local_spike_snr_min,
                spike_delta_snr_min=spike_delta_snr_min,
                noise_ref_mad=noise_ref_mad):
            out.append(cand_seg)
    return out


def merge_refined_focal_segments(base_segments: Sequence[dict],
                                 focal_subsegments: Sequence[dict]) -> List[dict]:
    """Return base segments plus refined focal events without focal double count."""
    if not focal_subsegments:
        return list(base_segments)
    refined_parent_keys = {
        (s.get("parent_start"), s.get("parent_end"), s.get("direction"))
        for s in focal_subsegments
    }
    merged: List[dict] = []
    for s in base_segments:
        key = (s.get("start"), s.get("end"), s.get("direction"))
        if key in refined_parent_keys:
            s2 = dict(s)
            s2["suppress_focal"] = True
            merged.append(s2)
        else:
            merged.append(s)
    merged.extend(focal_subsegments)
    return merged
