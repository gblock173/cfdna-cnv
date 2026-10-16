#!/usr/bin/env python3
"""Shared segment tier evaluation and plot styling for CNV caller.

Plot tier priority: focal > strong (baseline nf std/long) > weak (nf_capped std/long)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from focal_utils import focal_candidate_passes  # noqa: E402
except ImportError:
    from caller.focal_utils import focal_candidate_passes  # noqa: E402

try:
    from cfdna_utils import (  # noqa: E402
        cfdna_diluted_band_segments,
        is_cfdna_diluted_profile,
        seg_passes_cfdna_diluted,
    )
except ImportError:
    from caller.cfdna_utils import (  # noqa: E402
        cfdna_diluted_band_segments,
        is_cfdna_diluted_profile,
        seg_passes_cfdna_diluted,
    )

SEG_TIERS: List[dict] = [
    {"name": "focal",    "L_min": 5,   "L_max": 30,  "amp_min": 7.0, "peak_min": 10.0,
     "density_min": 0.50, "snr_min": 5.0, "peak_snr_min": 7.0},
    {"name": "standard", "L_min": 30,              "amp_min": 4.5, "peak_min": 5.0,
     "density_min": 0.30},
    {"name": "long",     "L_min": 300,             "amp_min": 2.0, "peak_min": 3.0,
     "density_min": 0.10},
]
ARM_TIERS: List[dict] = [
    {"name": "strong", "mean_min": 2.5, "frac_min": 0.30},
    {"name": "wide",   "mean_min": 1.2, "frac_min": 0.50},
]
SEG_SCALE_KEYS = ("amp_min", "peak_min")
ARM_SCALE_KEYS = ("mean_min",)
NOISE_REF_MAD = 1.0
NOISE_FACTOR_MAX_DEFAULT = 1.2

FOCAL_BAND_COLOR = "#9970ab"
STRONG_GAIN_BAND_COLOR = "#f4a582"
STRONG_LOSS_BAND_COLOR = "#92c5de"
WEAK_GAIN_BAND_COLOR = "#fddbc7"
WEAK_LOSS_BAND_COLOR = "#d1e5f0"
FOCAL_BAND_ALPHA = 0.55
STRONG_BAND_ALPHA = 0.35
WEAK_BAND_ALPHA = 0.60
PLOT_MIN_BAND_BIN_DEFAULT = 60


def noise_factor(raw_mad: float, ref_mad: float = NOISE_REF_MAD) -> float:
    if raw_mad is None or not float(raw_mad) == float(raw_mad) or raw_mad <= ref_mad:
        return 1.0
    return float(raw_mad / ref_mad)


def noise_factor_capped(nf: float, cap: float = NOISE_FACTOR_MAX_DEFAULT) -> float:
    if nf is None or not float(nf) == float(nf) or nf < 1.0:
        nf = 1.0
    return float(min(nf, cap))


def make_scaled_tiers(base_tiers: Sequence[dict], factor: float,
                      scale_keys: Sequence[str],
                      exempt_names: Sequence[str] = ("focal",)) -> List[dict]:
    if factor == 1.0:
        return [dict(t) for t in base_tiers]
    out = []
    for t in base_tiers:
        new_t = dict(t)
        if t["name"] not in exempt_names:
            for k in scale_keys:
                if k in new_t:
                    new_t[k] = new_t[k] * factor
        out.append(new_t)
    return out


def _seg_passes_tier(seg: dict, tier: dict, noise_factor_val: float = 1.0,
                     bg_mad: Optional[float] = None) -> bool:
    if tier["name"] == "focal" and seg.get("suppress_focal", False):
        return False
    if seg["length"] < tier["L_min"]:
        return False
    if "L_max" in tier and seg["length"] > tier["L_max"]:
        return False
    nf = 1.0 if tier["name"] == "focal" else noise_factor_val
    if not (seg["mean_abs_z"] >= tier["amp_min"]  * nf
            and seg["peak_abs_z"] >= tier["peak_min"] * nf
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


def seg_passes_focal(seg: dict, bg_mad: Optional[float] = None) -> bool:
    if seg.get("suppress_focal", False):
        return False
    if seg.get("from_subseg"):
        return focal_candidate_passes(seg, SEG_TIERS[0], bg_mad=bg_mad)
    return _seg_passes_tier(seg, SEG_TIERS[0], noise_factor_val=1.0, bg_mad=bg_mad)


def seg_passes_std_long(seg: dict, nf: float, bg_mad: Optional[float] = None) -> bool:
    tiers = make_scaled_tiers(SEG_TIERS, nf, SEG_SCALE_KEYS)
    for t in tiers:
        if t["name"] == "focal":
            continue
        if _seg_passes_tier(seg, t, noise_factor_val=1.0, bg_mad=bg_mad):
            return True
    return False


def segment_overlap_frac(a_start: int, a_end: int,
                         b_start: int, b_end: int) -> float:
    """Fraction of segment A bins that overlap segment B."""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    a_len = a_end - a_start + 1
    return float(overlap / a_len) if a_len > 0 else 0.0


def focal_embedded_in_broad_band(focal: dict,
                                 base_segments: Sequence[dict],
                                 nf: float,
                                 bg_mad: Optional[float] = None,
                                 nf_cap: float = NOISE_FACTOR_MAX_DEFAULT,
                                 overlap_min: float = 0.5) -> bool:
    """True when same-direction focal lies inside a strong/weak std/long band."""
    nf_c = noise_factor_capped(nf, cap=nf_cap)
    f0, f1 = int(focal["start"]), int(focal["end"])
    peak = int(focal.get("peak_bin", (f0 + f1) // 2))
    for base in base_segments:
        if base.get("from_subseg"):
            continue
        if base["direction"] != focal["direction"]:
            continue
        b0, b1 = int(base["start"]), int(base["end"])
        if not (b0 <= peak <= b1
                or segment_overlap_frac(f0, f1, b0, b1) >= overlap_min):
            continue
        if (seg_passes_std_long(base, nf, bg_mad=bg_mad)
                or seg_passes_std_long(base, nf_c, bg_mad=bg_mad)):
            return True
    return False


def filter_focal_not_in_broad_bands(
        focal_subsegments: Sequence[dict],
        base_segments: Sequence[dict],
        nf: float,
        bg_mad: Optional[float] = None,
        nf_cap: float = NOISE_FACTOR_MAX_DEFAULT,
        overlap_min: float = 0.5) -> List[dict]:
    """Drop refined focals already explained by same-direction strong/weak bands."""
    return [
        s for s in focal_subsegments
        if not focal_embedded_in_broad_band(
            s, base_segments, nf, bg_mad=bg_mad, nf_cap=nf_cap,
            overlap_min=overlap_min)
    ]


def seg_plot_tier(seg: dict, nf: float, bg_mad: Optional[float] = None,
                  nf_cap: float = NOISE_FACTOR_MAX_DEFAULT,
                  bin_z_mad: Optional[float] = None,
                  max_peak_abs_z: Optional[float] = None,
                  noisy_bin_mad: float = 1.81) -> Optional[str]:
    if seg_passes_focal(seg, bg_mad=bg_mad):
        return "focal"
    if seg_passes_std_long(seg, nf, bg_mad=bg_mad):
        return "strong"
    nf_c = noise_factor_capped(nf, cap=nf_cap)
    if seg_passes_std_long(seg, nf_c, bg_mad=bg_mad):
        return "weak"
    if (seg.get("from_diluted_merge")
            and bin_z_mad is not None and max_peak_abs_z is not None
            and is_cfdna_diluted_profile(bin_z_mad, nf, max_peak_abs_z,
                                         noisy_bin_mad=noisy_bin_mad)
            and seg_passes_cfdna_diluted(seg)):
        return "weak"
    return None


def augment_cfdna_diluted_segments(segments: Sequence[dict],
                                   z_smooth,
                                   nf: float,
                                   bin_z_mad: Optional[float],
                                   max_peak_abs_z: Optional[float],
                                   noisy_bin_mad: float = 1.81,
                                   bin_chrom=None,
                                   cfg: Optional[dict] = None) -> List[dict]:
    """Append merged long-span diluted bands for plot / tier evaluation."""
    if bin_z_mad is None or max_peak_abs_z is None:
        return list(segments)
    merged = cfdna_diluted_band_segments(
        segments, z_smooth, bin_z_mad, nf, max_peak_abs_z,
        noisy_bin_mad=noisy_bin_mad, bin_chrom=bin_chrom, cfg=cfg)
    if not merged:
        return list(segments)
    return list(segments) + merged


def summarize_weak_segments(segments: Sequence[dict], nf: float,
                            bg_mad: Optional[float] = None,
                            nf_cap: float = NOISE_FACTOR_MAX_DEFAULT) -> Dict[str, int]:
    nf_c = noise_factor_capped(nf, cap=nf_cap)
    n_gain = n_loss = 0
    for s in segments:
        if seg_passes_focal(s, bg_mad=bg_mad):
            continue
        if seg_passes_std_long(s, nf, bg_mad=bg_mad):
            continue
        if not seg_passes_std_long(s, nf_c, bg_mad=bg_mad):
            continue
        if s["direction"] > 0:
            n_gain += 1
        else:
            n_loss += 1
    return {"n_seg_weak_gain": int(n_gain), "n_seg_weak_loss": int(n_loss)}


def band_style(plot_tier: str, direction: int) -> Tuple[str, float]:
    if plot_tier == "focal":
        return FOCAL_BAND_COLOR, FOCAL_BAND_ALPHA
    if plot_tier == "strong":
        return ((STRONG_GAIN_BAND_COLOR if direction > 0 else STRONG_LOSS_BAND_COLOR),
                STRONG_BAND_ALPHA)
    if plot_tier == "weak":
        return ((WEAK_GAIN_BAND_COLOR if direction > 0 else WEAK_LOSS_BAND_COLOR),
                WEAK_BAND_ALPHA)
    return "#cccccc", 0.3


def arrow_style(plot_tier: str, direction: int) -> Tuple[str, str]:
    if plot_tier == "focal":
        return FOCAL_BAND_COLOR, "#5a3a78"
    if plot_tier == "strong":
        return (("#d73027", "#8a1a1a") if direction > 0 else ("#2166ac", "#143659"))
    if plot_tier == "weak":
        return (("#e08214", "#a05010") if direction > 0 else ("#4393c3", "#2166ac"))
    return "#888888", "#444444"


def draw_seg_band(ax, seg: dict, nf: float, bg_mad: Optional[float],
                  min_band_bin: int = PLOT_MIN_BAND_BIN_DEFAULT,
                  nf_cap: float = NOISE_FACTOR_MAX_DEFAULT,
                  bin_z_mad: Optional[float] = None,
                  max_peak_abs_z: Optional[float] = None,
                  noisy_bin_mad: float = 1.81) -> None:
    ptier = seg_plot_tier(seg, nf, bg_mad=bg_mad, nf_cap=nf_cap,
                          bin_z_mad=bin_z_mad, max_peak_abs_z=max_peak_abs_z,
                          noisy_bin_mad=noisy_bin_mad)
    if ptier is None:
        return
    L = seg["end"] - seg["start"] + 1
    L_vis = L if ptier == "focal" else max(L, min_band_bin)
    pad = (L_vis - L) / 2
    x0 = seg["start"] - 0.5 - pad
    x1 = seg["end"]   + 0.5 + pad
    color, alpha = band_style(ptier, seg["direction"])
    ax.axvspan(x0, x1, color=color, alpha=alpha, zorder=0)


def draw_seg_arrows(ax, segments: Sequence[dict], nf: float, bg_mad: Optional[float],
                    ylim: Tuple[float, float], y_offset_frac: float = 0.06,
                    nf_cap: float = NOISE_FACTOR_MAX_DEFAULT) -> None:
    if not segments:
        return
    y_top, y_bot = ylim[1], ylim[0]
    rng = y_top - y_bot
    y_up = y_top - rng * y_offset_frac
    y_dn = y_bot + rng * y_offset_frac
    for s in segments:
        ptier = seg_plot_tier(s, nf, bg_mad=bg_mad, nf_cap=nf_cap)
        if ptier is None:
            continue
        xc = float(s.get("peak_bin", 0.5 * (s["start"] + s["end"])))
        face, edge = arrow_style(ptier, s["direction"])
        marker = "^" if s["direction"] > 0 else "v"
        y = y_up if s["direction"] > 0 else y_dn
        ms = 90 if ptier == "focal" else 70
        ax.scatter([xc], [y], marker=marker, s=ms, c=face,
                   edgecolors=edge, linewidths=0.8, zorder=6, clip_on=False)


def legend_elements(z_call_thr: float = 3.0, show_focal: bool = True) -> List:
    elems = [
        Patch(facecolor="#dddddd", label="Healthy Train Q5-Q95"),
        Patch(facecolor=STRONG_GAIN_BAND_COLOR, alpha=STRONG_BAND_ALPHA,
              label="sig strong gain (baseline std/long)"),
        Patch(facecolor=STRONG_LOSS_BAND_COLOR, alpha=STRONG_BAND_ALPHA,
              label="sig strong loss (baseline std/long)"),
        Patch(facecolor=WEAK_GAIN_BAND_COLOR, alpha=WEAK_BAND_ALPHA,
              label="sig weak gain (nf cap std/long)"),
        Patch(facecolor=WEAK_LOSS_BAND_COLOR, alpha=WEAK_BAND_ALPHA,
              label="sig weak loss (nf cap std/long)"),
    ]
    if show_focal:
        elems.append(Patch(facecolor=FOCAL_BAND_COLOR, alpha=FOCAL_BAND_ALPHA,
                           label="sig focal seg"))
    elems += [
        Line2D([0], [0], color="black", marker="o", ms=5, lw=0, label="within band"),
        Line2D([0], [0], color="#d73027", marker="o", ms=5, lw=0,
               label=f"gain bin (z>{z_call_thr:.0f})"),
        Line2D([0], [0], color="#2166ac", marker="o", ms=5, lw=0,
               label=f"loss bin (z<-{z_call_thr:.0f})"),
        Line2D([0], [0], color="#d73027", marker="^", ms=8, lw=0,
               markeredgecolor="#8a1a1a", label="seg arrow: strong gain"),
        Line2D([0], [0], color="#2166ac", marker="v", ms=8, lw=0,
               markeredgecolor="#143659", label="seg arrow: strong loss"),
        Line2D([0], [0], color="#e08214", marker="^", ms=8, lw=0,
               markeredgecolor="#a05010", label="seg arrow: weak gain"),
        Line2D([0], [0], color="#4393c3", marker="v", ms=8, lw=0,
               markeredgecolor="#2166ac", label="seg arrow: weak loss"),
    ]
    if show_focal:
        elems.append(
            Line2D([0], [0], color=FOCAL_BAND_COLOR, marker="^", ms=8, lw=0,
                   markeredgecolor="#5a3a78", label="seg arrow: focal"))
    return elems
