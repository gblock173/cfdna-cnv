"""Exercise the distributed synthetic example and configuration portability."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from caller.utils.config_loader import load_config

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd, env):
    result = subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_relative_config_paths_resolve_from_config_directory(tmp_path, monkeypatch):
    folder = tmp_path / "configuration"
    folder.mkdir()
    config = folder / "test.yaml"
    config.write_text("paths:\n  meta_xlsx: ../data/samples.xlsx\n  meta_sheet: samples\n")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(config)
    assert cfg["paths"]["meta_xlsx"] == str(tmp_path / "data/samples.xlsx")
    assert cfg["paths"]["meta_sheet"] == "samples"


def test_release_manifest_excludes_data_and_binary_files():
    files = (ROOT / "release_files.txt").read_text().splitlines()
    allowed = {".py", ".r", ".md", ".yaml", ".yml", ".txt", ".json", ""}
    assert files
    for name in files:
        path = ROOT / name
        assert not path.is_symlink()
        assert path.suffix.lower() in allowed
        assert not name.startswith(("work/", "data/", "references/", "results/"))
        path.read_text(encoding="utf-8")


@pytest.mark.parametrize("depth", ["20G", "100G", "WGS"])
def test_synthetic_pipeline_from_another_directory(tmp_path, depth):
    env = os.environ.copy()
    env.pop("CFDNA_CNV_CONFIG", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    run(ROOT / "examples/synthetic/make_inputs.py", "--output", inputs,
        "--depth", depth, cwd=tmp_path, env=env)
    with np.load(inputs / "reference.npz", allow_pickle=False) as ref:
        assert int(ref["pon_n_effective"]) == 0
        assert str(ref["depth_label"]) == depth
    command = [ROOT / "scripts/run_pipeline.py", "--config", inputs / "config.yaml",
               "--aid", "example", "--qdnaseq", inputs / "sample.log2.tsv",
               "--out-dir", output]
    run(*command, "--plan-only", cwd=tmp_path, env=env)
    assert not output.exists()
    log = run(*command, *(["--profile"] if depth == "100G" else []), cwd=tmp_path, env=env)
    actual = pd.read_csv(output / "label_pred.tsv", sep="\t")
    expected = pd.DataFrame({"AnalysisID": ["example"], "label": ["CNV+_strong"], "pred": [1]})
    pd.testing.assert_frame_equal(actual[expected.columns], expected)
    assert actual.depth.tolist() == [depth]
    assert actual.version.tolist() == [(ROOT / "VERSION").read_text().strip()]
    if depth == "100G":
        assert "MISMATCH" not in log
        assert len(list((output / "profile").glob("*.png"))) == 1
    repeated = subprocess.run([sys.executable, *map(str, command)], cwd=tmp_path, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert repeated.returncode != 0
    assert "must be empty" in repeated.stdout
