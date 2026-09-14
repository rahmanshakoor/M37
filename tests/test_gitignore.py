"""The repository must refuse patient data by construction.

Every path a stage could write with genotypes in it, and every input file, must be
ignored — and ordinary source must not be. Uses git's own matcher so the test cannot
drift from what git actually does.
"""

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MUST_IGNORE = [
    "data/proband.vcf.gz",
    "data/proband.vcf.gz.tbi",
    "anything/at/all.vcf",
    "x.bcf",
    "reads.fastq.gz",
    "reads.bam",
    "case.yaml",
    "case.proband.yaml",
    "work/run1/01_ingest/variants.tsv.gz",
    "work/run1/01_ingest/manifest.json",
    "cache/vep/abc.json",
    "ref/GRCh38.fa",
    "results.tsv",
    "submission.csv",
    "shortlist.parquet",
    "Challenge_Clinical_Phenotype_1.docx",
    "report.pdf",
]

MUST_KEEP = [
    "case.example.yaml",
    "src/engine/ingest.py",
    "configs/panel.yaml",
    "configs/panel_coords.grch38.json",
    "configs/panel.grch38.bed",
    "tests/test_ingest.py",
    "README.md",
    "pyproject.toml",
]


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", path], check=False)
    return r.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _needs_git_repo():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_patient_data_paths_are_ignored(path):
    assert _ignored(path), f"{path} would be committed"


@pytest.mark.parametrize("path", MUST_KEEP)
def test_source_paths_are_not_ignored(path):
    assert not _ignored(path), f"{path} is ignored but should be tracked"
