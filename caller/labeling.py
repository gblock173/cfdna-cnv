"""
labeling.py
7-label classification logic (shared module).

Labels (priority order):
  1. CNV+_strong  = seg_pos AND arm_pos
  2. CNV+_weak    = seg_weak (nf-capped OR arm-concordant diluted gain)
  3. QC_noisy_weak = weak signal suppressed by the Train-OOF noise gate
  4. CNV_suspect  = seg_pos OR arm_pos OR raw diluted band
  5. Focal_only   = focal segment only, no other signal
  6. HN           = high noise (bin_z_MAD >= threshold)
  7. Normal       = none of the above
"""

LABEL_ORDER = ["CNV+_strong", "CNV+_weak", "QC_noisy_weak", "CNV_suspect", "Focal_only", "HN", "Normal"]

LABEL_COLOR = {
    "CNV+_strong": "#1a9850",
    "CNV+_weak":   "#fdae61",
    "QC_noisy_weak": "#8c6d31",
    "CNV_suspect": "#f6e36f",
    "Focal_only":  "#9970ab",
    "HN":          "#d73027",
    "Normal":      "#bdbdbd",
}


def is_seg_pos(row):
    """seg-level positivity = standard or long tier (focal excluded)."""
    return ((row.get("n_seg_standard", 0) >= 1)
         or (row.get("n_seg_long",     0) >= 1))


def is_arm_pos(row):
    """arm-level strong positivity = strong/wide tier arm >= 1."""
    return ((row.get("n_arms_strong", 0) >= 1)
         or (row.get("n_arms_wide",   0) >= 1))


def is_seg_weak(row, gain_only=False):
    """weak segment = nf-capped std/long tier OR arm-concordant diluted gain.

    Diluted concordance channel is gain-only by design (no n_diluted_conc_loss).
    The nf-capped segment tier includes the loss channel (n_seg_weak_loss) by
    default. Set gain_only=True for the 'gain-only weak' version, which drops
    the loss segment channel; very noisy loss-only samples then fall to HN.
    Non-concordant diluted bands are demoted to CNV_suspect via has_diluted_raw.
    """
    weak = ((row.get("n_seg_weak_gain", 0) >= 1)
         or (row.get("n_diluted_conc_gain", 0) >= 1))
    if not gain_only:
        weak = weak or (row.get("n_seg_weak_loss", 0) >= 1)
    return weak


def has_diluted_raw(row):
    """Any diluted band (including non-concordant) exists."""
    return ((row.get("n_diluted_gain", 0) >= 1)
         or (row.get("n_diluted_loss", 0) >= 1))


def classify(row, noisy_bin_mad=1.81, weak_gain_only=False,
             weak_noise_mad=None, weak_rescue_peak_snr=None,
             weak_noise_label="QC_noisy_weak"):
    """Assign a label based on segment/arm/noise metrics.

    Parameters
    ----------
    row : dict-like
        Must have keys: n_seg_standard, n_seg_long, n_arms_strong, n_arms_wide,
        n_seg_weak_gain, n_seg_weak_loss, n_diluted_conc_gain,
        n_diluted_gain, n_diluted_loss, n_seg_focal, bin_z_MAD.
    noisy_bin_mad : float
        HN threshold (depth-specific Q95 of healthy Train bin_z_MAD).
    weak_gain_only : bool
        Default False = current operating behavior (weak segment tier includes
        loss). True = 'gain-only weak' version: drop the weak loss channel.
    weak_noise_mad : float or None
        Train-only OOF-selected bin_z_MAD threshold for suppressing weak calls.
        None preserves the legacy behavior.
    weak_rescue_peak_snr : float or None
        Rescue threshold for max_seg_peak_abs_z / max(noise_factor, 1).
    weak_noise_label : str
        Label assigned to suppressed weak calls.
    """
    seg_pos = is_seg_pos(row)
    arm_pos = is_arm_pos(row)
    seg_wk  = is_seg_weak(row, gain_only=weak_gain_only)
    if seg_pos and arm_pos:
        return "CNV+_strong"
    if seg_wk:
        if weak_noise_mad is not None:
            bin_mad = float(row.get("bin_z_MAD", 0))
            noise_factor = float(row.get("noise_factor", 1))
            if noise_factor != noise_factor or noise_factor < 1:
                noise_factor = 1.0
            peak_snr = float(row.get("max_seg_peak_abs_z", 0)) / noise_factor
            rescued = (weak_rescue_peak_snr is not None
                       and peak_snr >= float(weak_rescue_peak_snr))
            if bin_mad >= float(weak_noise_mad) and not rescued:
                return weak_noise_label
        return "CNV+_weak"
    if seg_pos or arm_pos or has_diluted_raw(row):
        return "CNV_suspect"
    if row.get("n_seg_focal", 0) >= 1:
        return "Focal_only"
    if row["bin_z_MAD"] >= noisy_bin_mad:
        return "HN"
    return "Normal"
