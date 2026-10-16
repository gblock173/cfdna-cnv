# cfDNA CNV caller

Rule-based CNV calling for hg38 autosomal QDNAseq log2 ratios in 100 kb bins.
Author: **Yujin Jeon**. Version: **1.0.1**. Separate parameter presets cover `20G`, `100G`, and `WGS`.
The workflow anchors each sample, standardizes against a panel of normals (PoN),
computes segment/arm and noise metrics, and assigns one of seven labels.

| Label | Binary prediction |
| --- | --- |
| CNV+_strong | 1 |
| CNV+_weak | 1 |
| QC_noisy_weak | 0 |
| CNV_suspect | 0 |
| Focal_only | 0 |
| HN | 0 |
| Normal | 0 |

This code-only release contains no sample data files, clinical records,
production references, cohort results, private server paths or credentials.
An optional generator creates a synthetic demonstration locally when requested.
It does not read any external measurements. The demonstration establishes
software execution only, not analytical or clinical performance. Build a
suitable reference from your own controls before using real measurements.

## Installation

Use Python 3.10 or newer in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.
Python scripts also work without the shell activation if the environment's
Python executable is used explicitly.

## Generate a local example

Run from the repository root:

```bash
python examples/synthetic/make_inputs.py --output work/synthetic --depth 100G
python scripts/run_pipeline.py \
  --config work/synthetic/config.yaml \
  --aid example \
  --qdnaseq work/synthetic/sample.log2.tsv \
  --out-dir work/synthetic-run --profile
```

Expected: `example`, label `CNV+_strong`, prediction `1`.
The final table is `work/synthetic-run/label_pred.tsv`; detailed metrics and
labels are in `calling/`, and an optional profile is in `profile/`.
Use fresh output directories when repeating a run. `--plan-only` checks paths
and prints commands without running analysis.

The generator uses NumPy seed 7421, public hg38 autosomal chromosome lengths,
100 kb bins, zero-mean Gaussian noise with standard deviation
`0.04 / 0.6744897501960817`, and a log2 gain of 0.45 over chromosome 1,
20,000,001-65,000,000 (1-based inclusive). The reference uses a zero median
and constant MAD 0.04 with zero human controls. These values define a simulation;
they are not measurements from samples. All generated data, metadata, references,
configuration and results remain under the ignored `work/` directory.
Change `--depth` and output directories to exercise the other parameter presets.

## Your own data

1. Copy `configs/config_100G.example.yaml` to `configs/config_100G.local.yaml`
   (or select `20G`/`WGS`). Relative paths resolve from the configuration file.
2. Set `paths.meta_xlsx`, `meta_sheet`, `qdnaseq_matrix`, and `pon_reference`.
   The control matrix is a tab-separated table with a `feature` column and
   one numeric log2 column per sample. Features use `1:1-100000` form without
   a `chr` prefix. Preserve QDNAseq coordinates and reference feature ordering.
3. Metadata is an XLSX sheet containing `AnalysisID`, `CancerType`, `Stage`,
   and `Split_AgeSexStage`. Use `CancerType=healthy` and
   `Split_AgeSexStage=Train` for reference controls. Optional profiles also
   require `Age` and `Sex` columns; missing values are allowed.
4. Build a reference using your own healthy training controls:

```bash
python scripts/sample_pipeline/00_build_pon_reference.py \
  --depth 100G --config configs/config_100G.local.yaml
```

At least 20 controls must remain after noise filtering; this is a software
minimum, not a recommendation for a validated reference design. Match reference
and sample genome build, bin definitions, assay/depth and processing.

5. Run `scripts/run_pipeline.py` with your config, sample ID, QDNAseq file and
   a new output directory. A single-sample QDNAseq file must contain `feature`,
   `chromosome`, `start`, `end`, and exactly one numeric signal column.

The runner executes locally and requires no batch scheduler, SSH or GPU.
The supplied rule thresholds are configurable; no performance claim is made.

## Optional BAM input

Install R and the dependencies in `environment.yml`, including QDNAseq,
QDNAseq.hg38, optparse, Biobase, GenomeInfoDb and DNAcopy. Supply your own
coordinate-sorted, paired-end BAM and appropriate index. Replace `--qdnaseq`
with `--bam sample.bam`; use `--rscript` if Rscript is not on PATH.
The extractor uses hg38 100 kb bins. No example BAM is distributed.

## Tests and release contents

```bash
python -m pytest -q
sha256sum -c SHA256SUMS
```

Tests cover classification and export logic, relative configuration paths,
and the full synthetic input-to-prediction workflow for all three presets.
The optional BAM conversion requires an external BAM and its R environment.
Only paths listed in `release_files.txt` belong to this release. Local data,
references, analysis outputs, credentials and caches must not be added.

## License

MIT. Copyright (c) 2026 Yujin Jeon. See `LICENSE`.
