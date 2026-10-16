#!/usr/bin/env python3
"""
hg38 chromosome arm coordinates based on UCSC centromere annotations.

Each arm is defined as (chrom, arm_start, arm_end) where:
  - p arm: chrom_start (0) to centromere_start
  - q arm: centromere_end to chrom_end

Centromere positions sourced from UCSC hg38 cytoBand.txt (acen bands).
Chromosome sizes from hg38 chrom.sizes.
"""

from typing import Dict, List, NamedTuple

import numpy as np


class ArmCoord(NamedTuple):
    chrom: str
    start: int
    end: int
    arm: str  # e.g. "1p", "1q"


# hg38 centromere midpoints (acen band start/end from UCSC cytoBand)
# Format: chrom -> (centromere_start, centromere_end)
# Values from UCSC hg38 cytoBand.txt, acen regions
HG38_CENTROMERE: Dict[str, tuple] = {
    "chr1":  (121700000, 125100000),
    "chr2":  (91800000,  96000000),
    "chr3":  (87800000,  94000000),
    "chr4":  (48200000,  51800000),
    "chr5":  (46100000,  51400000),
    "chr6":  (58500000,  62600000),
    "chr7":  (58100000,  62100000),
    "chr8":  (43200000,  47200000),
    "chr9":  (42200000,  45500000),
    "chr10": (38000000,  41600000),
    "chr11": (51000000,  55800000),
    "chr12": (33200000,  37800000),
    "chr13": (16500000,  18900000),
    "chr14": (16100000,  18200000),
    "chr15": (17500000,  20500000),
    "chr16": (35300000,  38400000),
    "chr17": (22700000,  27400000),
    "chr18": (15400000,  21500000),
    "chr19": (24400000,  28100000),
    "chr20": (25700000,  30400000),
    "chr21": (10900000,  13000000),
    "chr22": (13700000,  17400000),
    "chrX":  (60600000,  65800000),
    "chrY":  (10300000,  10600000),
}

# hg38 chromosome sizes
HG38_CHROM_SIZES: Dict[str, int] = {
    "chr1":  248956422,
    "chr2":  242193529,
    "chr3":  198295559,
    "chr4":  190214555,
    "chr5":  181538259,
    "chr6":  170805979,
    "chr7":  159345973,
    "chr8":  145138636,
    "chr9":  138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX":  156040895,
    "chrY":  57227415,
}

# Autosomes + X; Y excluded by default (small q arm, often unreliable)
AUTOSOME_CHROMS = [f"chr{i}" for i in range(1, 23)]
SEX_CHROMS = ["chrX"]
DEFAULT_CHROMS = AUTOSOME_CHROMS + SEX_CHROMS


def build_arm_coords(chroms: List[str] = None) -> List[ArmCoord]:
    """
    Returns a list of ArmCoord for each arm (p and q) of the given chromosomes.

    Chromosomes without a defined centromere in HG38_CENTROMERE are skipped.
    For acrocentric chromosomes (13, 14, 15, 21, 22), the p arm is very short
    and may have no bins; they are still included for completeness.
    """
    if chroms is None:
        chroms = DEFAULT_CHROMS

    coords: List[ArmCoord] = []
    for chrom in chroms:
        if chrom not in HG38_CENTROMERE:
            continue
        cent_start, cent_end = HG38_CENTROMERE[chrom]
        chrom_size = HG38_CHROM_SIZES.get(chrom, 0)
        chrom_num = chrom.replace("chr", "")

        # p arm: 0 → centromere_start
        if cent_start > 0:
            coords.append(ArmCoord(
                chrom=chrom,
                start=0,
                end=cent_start,
                arm=f"{chrom_num}p",
            ))

        # q arm: centromere_end → chrom_end
        if chrom_size > cent_end:
            coords.append(ArmCoord(
                chrom=chrom,
                start=cent_end,
                end=chrom_size,
                arm=f"{chrom_num}q",
            ))

    return coords


def get_arm_names(chroms: List[str] = None) -> List[str]:
    """Returns ordered list of arm names (e.g. ['1p', '1q', ..., 'Xp', 'Xq'])."""
    return [ac.arm for ac in build_arm_coords(chroms)]


# Precomputed defaults for import convenience
ARM_COORDS: List[ArmCoord] = build_arm_coords()
ARM_NAMES: List[str] = get_arm_names()


def make_bin_arm(chrs_sorted, starts_sorted) -> np.ndarray:
    """Return object array of arm label per bin, e.g. '1p', '1q'.

    Centromere bins (centromere_start <= start < centromere_end) are labeled None.
    Used by extract_segments() to break hysteresis segments at p/q arm boundaries.

    Parameters
    ----------
    chrs_sorted  : array-like of str, chromosome name per bin (no 'chr' prefix, e.g. '5')
    starts_sorted: array-like of int, bin start position in bp
    """
    cent_map = {
        c.replace("chr", ""): (cs, ce)
        for c, (cs, ce) in HG38_CENTROMERE.items()
    }
    n = len(chrs_sorted)
    arm = np.empty(n, dtype=object)
    for i in range(n):
        c = str(chrs_sorted[i])
        s = int(starts_sorted[i])
        if c not in cent_map:
            arm[i] = f"{c}p"
            continue
        cs, ce = cent_map[c]
        if s < cs:
            arm[i] = f"{c}p"
        elif s >= ce:
            arm[i] = f"{c}q"
        else:
            arm[i] = None  # centromere region — treated as gap in extract_segments
    return arm
