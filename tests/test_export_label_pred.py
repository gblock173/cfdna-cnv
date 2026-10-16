import pandas as pd
import pytest

from scripts.final_pipeline.export_label_pred import (
    build_label_pred,
    write_atomic_tsv,
)


def test_build_label_pred_maps_only_strong_and_weak_to_one(tmp_path):
    source = tmp_path / "labels.tsv"
    pd.DataFrame({
        "AnalysisID": ["A", "B", "C", "D"],
        "label": ["CNV+_strong", "CNV+_weak", "QC_noisy_weak", "Normal"],
        "CancerType": ["cancer", "cancer", "healthy", "healthy"],
    }).to_csv(source, sep="\t", index=False)
    result = build_label_pred([("100G", source)], "fixed-v1")
    assert result["pred"].tolist() == [1, 1, 0, 0]
    assert result["version"].unique().tolist() == ["fixed-v1"]
    assert result["depth"].unique().tolist() == ["100G"]


def test_build_label_pred_rejects_unknown_label(tmp_path):
    source = tmp_path / "labels.tsv"
    pd.DataFrame({"AnalysisID": ["A"], "label": ["unexpected"]}).to_csv(
        source, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="알 수 없는 label"):
        build_label_pred([("20G", source)], "fixed-v1")


def test_build_label_pred_rejects_duplicate_depth(tmp_path):
    source = tmp_path / "labels.tsv"
    pd.DataFrame({"AnalysisID": ["A"], "label": ["Normal"]}).to_csv(
        source, sep="\t", index=False
    )
    with pytest.raises(ValueError, match="depth 입력 중복"):
        build_label_pred([("WGS", source), ("WGS", source)], "fixed-v1")


def test_write_atomic_tsv(tmp_path):
    output = tmp_path / "nested" / "result.tsv"
    frame = pd.DataFrame({"AnalysisID": ["A"], "pred": [0]})
    write_atomic_tsv(frame, output)
    loaded = pd.read_csv(output, sep="\t")
    assert loaded.to_dict("records") == [{"AnalysisID": "A", "pred": 0}]
    assert not output.with_suffix(".tsv.tmp").exists()
