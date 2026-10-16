"""
test_classify.py — 7-label classify() 단위 테스트.

각 label + 경계 우선순위 + noisy_bin_mad 파라미터화 테스트.
"""
import pytest
from caller.labeling import classify, is_seg_pos, is_arm_pos, is_seg_weak, has_diluted_raw


def _base_row(**overrides):
    """All-zero 기본 row (Normal이 되어야 함). overrides로 특정 값만 변경."""
    row = {
        "n_seg_standard": 0, "n_seg_long": 0,
        "n_arms_strong": 0, "n_arms_wide": 0,
        "n_seg_weak_gain": 0, "n_seg_weak_loss": 0,
        "n_diluted_conc_gain": 0,
        "n_diluted_gain": 0, "n_diluted_loss": 0,
        "n_seg_focal": 0,
        "bin_z_MAD": 0.5,
    }
    row.update(overrides)
    return row


# ── Label 1: CNV+_strong ──────────────────────────────────────────

class TestCNVStrong:
    def test_standard_seg_and_strong_arm(self):
        row = _base_row(n_seg_standard=1, n_arms_strong=1)
        assert classify(row) == "CNV+_strong"

    def test_long_seg_and_wide_arm(self):
        row = _base_row(n_seg_long=1, n_arms_wide=1)
        assert classify(row) == "CNV+_strong"

    def test_standard_seg_and_wide_arm(self):
        row = _base_row(n_seg_standard=2, n_arms_wide=1)
        assert classify(row) == "CNV+_strong"

    def test_strong_beats_weak_when_both_present(self):
        """seg_pos + arm_pos + seg_weak 모두 있으면 strong이 우선."""
        row = _base_row(n_seg_standard=1, n_arms_strong=1, n_seg_weak_gain=1)
        assert classify(row) == "CNV+_strong"


# ── Label 2: CNV+_weak ────────────────────────────────────────────

class TestCNVWeak:
    def test_weak_gain_only(self):
        row = _base_row(n_seg_weak_gain=1)
        assert classify(row) == "CNV+_weak"

    def test_weak_loss_only(self):
        row = _base_row(n_seg_weak_loss=1)
        assert classify(row) == "CNV+_weak"

    def test_diluted_conc_gain_only(self):
        row = _base_row(n_diluted_conc_gain=1)
        assert classify(row) == "CNV+_weak"

    def test_weak_beats_suspect(self):
        """seg_weak 이 raw diluted 보다 우선."""
        row = _base_row(n_seg_weak_gain=1, n_diluted_gain=1)
        assert classify(row) == "CNV+_weak"

    def test_weak_without_arm_pos(self):
        """arm_pos 없이 seg_weak만 있으면 weak (strong 아님)."""
        row = _base_row(n_seg_weak_gain=1, n_arms_strong=0)
        assert classify(row) == "CNV+_weak"


# ── Label 3: CNV_suspect ──────────────────────────────────────────

class TestCNVSuspect:
    def test_seg_pos_only(self):
        row = _base_row(n_seg_standard=1)
        assert classify(row) == "CNV_suspect"

    def test_arm_pos_only(self):
        row = _base_row(n_arms_strong=1)
        assert classify(row) == "CNV_suspect"

    def test_long_seg_no_arm(self):
        row = _base_row(n_seg_long=1)
        assert classify(row) == "CNV_suspect"

    def test_wide_arm_no_seg(self):
        row = _base_row(n_arms_wide=1)
        assert classify(row) == "CNV_suspect"

    def test_raw_diluted_gain_only(self):
        """non-concordant diluted → suspect."""
        row = _base_row(n_diluted_gain=1)
        assert classify(row) == "CNV_suspect"

    def test_raw_diluted_loss_only(self):
        row = _base_row(n_diluted_loss=1)
        assert classify(row) == "CNV_suspect"


# ── Label 4: Focal_only ──────────────────────────────────────────

class TestFocalOnly:
    def test_focal_seg_only(self):
        row = _base_row(n_seg_focal=1)
        assert classify(row) == "Focal_only"

    def test_focal_multiple(self):
        row = _base_row(n_seg_focal=3)
        assert classify(row) == "Focal_only"

    def test_focal_with_strong_becomes_strong(self):
        """focal + seg_pos + arm_pos → CNV+_strong (not Focal_only)."""
        row = _base_row(n_seg_focal=1, n_seg_standard=1, n_arms_strong=1)
        assert classify(row) == "CNV+_strong"

    def test_focal_with_seg_pos_becomes_suspect(self):
        """focal + seg_pos (no arm) → CNV_suspect."""
        row = _base_row(n_seg_focal=1, n_seg_standard=1)
        assert classify(row) == "CNV_suspect"


# ── Label 5: HN (High Noise) ─────────────────────────────────────

class TestHN:
    def test_high_noise_default_threshold(self):
        row = _base_row(bin_z_MAD=1.81)
        assert classify(row) == "HN"

    def test_just_below_threshold(self):
        row = _base_row(bin_z_MAD=1.80)
        assert classify(row) == "Normal"

    def test_custom_threshold_20G(self):
        row = _base_row(bin_z_MAD=1.40)
        assert classify(row, noisy_bin_mad=1.40) == "HN"

    def test_custom_threshold_WGS(self):
        row = _base_row(bin_z_MAD=1.10)
        assert classify(row, noisy_bin_mad=1.10) == "HN"

    def test_noisy_but_has_signal_not_HN(self):
        """HN 노이즈 수준이지만 seg_pos 있으면 CNV_suspect (HN보다 우선)."""
        row = _base_row(bin_z_MAD=2.5, n_seg_standard=1)
        assert classify(row) == "CNV_suspect"

    def test_noisy_with_focal_becomes_focal(self):
        """noisy지만 focal 있으면 Focal_only (HN보다 우선)."""
        row = _base_row(bin_z_MAD=2.0, n_seg_focal=1)
        assert classify(row) == "Focal_only"


# ── Label 6: Normal ──────────────────────────────────────────────

class TestNormal:
    def test_all_zero(self):
        row = _base_row()
        assert classify(row) == "Normal"

    def test_low_noise(self):
        row = _base_row(bin_z_MAD=0.93)
        assert classify(row) == "Normal"


# ── Helper function tests ────────────────────────────────────────

class TestHelpers:
    def test_is_seg_pos_standard(self):
        assert is_seg_pos({"n_seg_standard": 1, "n_seg_long": 0}) is True

    def test_is_seg_pos_long(self):
        assert is_seg_pos({"n_seg_standard": 0, "n_seg_long": 1}) is True

    def test_is_seg_pos_none(self):
        assert is_seg_pos({"n_seg_standard": 0, "n_seg_long": 0}) is False

    def test_is_arm_pos_strong(self):
        assert is_arm_pos({"n_arms_strong": 1, "n_arms_wide": 0}) is True

    def test_is_arm_pos_wide(self):
        assert is_arm_pos({"n_arms_strong": 0, "n_arms_wide": 1}) is True

    def test_is_arm_pos_none(self):
        assert is_arm_pos({"n_arms_strong": 0, "n_arms_wide": 0}) is False

    def test_is_seg_weak_gain(self):
        assert is_seg_weak({"n_seg_weak_gain": 1, "n_seg_weak_loss": 0, "n_diluted_conc_gain": 0}) is True

    def test_is_seg_weak_diluted_conc(self):
        assert is_seg_weak({"n_seg_weak_gain": 0, "n_seg_weak_loss": 0, "n_diluted_conc_gain": 1}) is True

    def test_has_diluted_raw_gain(self):
        assert has_diluted_raw({"n_diluted_gain": 1, "n_diluted_loss": 0}) is True

    def test_has_diluted_raw_loss(self):
        assert has_diluted_raw({"n_diluted_gain": 0, "n_diluted_loss": 1}) is True

    def test_has_diluted_raw_none(self):
        assert has_diluted_raw({"n_diluted_gain": 0, "n_diluted_loss": 0}) is False


# ── Priority order tests (종합) ──────────────────────────────────

class TestPriorityOrder:
    """분류 우선순위: strong > weak > suspect > focal > HN > Normal"""

    def test_all_signals_present(self):
        """모든 시그널이 동시에 있으면 strong이 이김."""
        row = _base_row(
            n_seg_standard=1, n_seg_long=1,
            n_arms_strong=1, n_arms_wide=1,
            n_seg_weak_gain=1, n_seg_weak_loss=1,
            n_diluted_conc_gain=1,
            n_diluted_gain=1, n_diluted_loss=1,
            n_seg_focal=3,
            bin_z_MAD=3.0,
        )
        assert classify(row) == "CNV+_strong"

    def test_weak_plus_diluted_plus_focal_plus_hn(self):
        """weak > suspect > focal > HN."""
        row = _base_row(
            n_seg_weak_gain=1,
            n_diluted_gain=1,
            n_seg_focal=1,
            bin_z_MAD=2.5,
        )
        assert classify(row) == "CNV+_weak"

    def test_suspect_plus_focal_plus_hn(self):
        """suspect > focal > HN."""
        row = _base_row(n_diluted_gain=1, n_seg_focal=1, bin_z_MAD=2.5)
        assert classify(row) == "CNV_suspect"

    def test_focal_plus_hn(self):
        """focal > HN."""
        row = _base_row(n_seg_focal=1, bin_z_MAD=2.5)
        assert classify(row) == "Focal_only"


# ── Train-OOF weak-noise gate ────────────────────────────────────

class TestWeakNoiseTrainOOF:
    def test_disabled_by_default_preserves_legacy_weak_call(self):
        row = _base_row(
            n_seg_weak_gain=1,
            bin_z_MAD=2.0,
            noise_factor=2.0,
            max_seg_peak_abs_z=2.0,
        )
        assert classify(row) == "CNV+_weak"

    def test_noisy_weak_is_suppressed_without_peak_snr_rescue(self):
        row = _base_row(
            n_seg_weak_gain=1,
            bin_z_MAD=1.5,
            noise_factor=2.0,
            max_seg_peak_abs_z=6.0,
        )
        assert classify(
            row, weak_noise_mad=1.4, weak_rescue_peak_snr=3.5
        ) == "QC_noisy_weak"

    def test_peak_snr_rescues_noisy_weak(self):
        row = _base_row(
            n_seg_weak_gain=1,
            bin_z_MAD=1.5,
            noise_factor=2.0,
            max_seg_peak_abs_z=7.0,
        )
        assert classify(
            row, weak_noise_mad=1.4, weak_rescue_peak_snr=3.5
        ) == "CNV+_weak"

    def test_strong_call_is_never_suppressed(self):
        row = _base_row(
            n_seg_standard=1,
            n_arms_strong=1,
            n_seg_weak_gain=1,
            bin_z_MAD=3.0,
            noise_factor=3.0,
            max_seg_peak_abs_z=1.0,
        )
        assert classify(
            row, weak_noise_mad=1.4, weak_rescue_peak_snr=3.5
        ) == "CNV+_strong"
