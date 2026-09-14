from pathlib import Path

from engine.retrieve.intervals import IntervalIndex


def test_bed_membership_is_half_open_and_one_based_aware(tmp_path: Path):
    bed = tmp_path / "r.bed"
    bed.write_text("15\t100\t200\tA\nchr15\t150\t300\tB\n1\t10\t20\tC\n1_KI270706v1_random\t0\t5\tZ\n")
    idx = IntervalIndex.from_bed(bed)
    assert idx.n_intervals() == 2  # 15 intervals merged, random contig dropped
    assert idx.total_bp() == 200 + 10
    # BED [100,300) covers 1-based positions 101..300
    assert not idx.contains("15", 100)
    assert idx.contains("15", 101)
    assert idx.contains("15", 300)
    assert not idx.contains("15", 301)
    assert idx.lookup("15", 250) == "A,B"
    assert idx.lookup("2", 5) is None
    assert idx.contains("1", 11) and not idx.contains("1", 21)


def test_overlap_uses_reference_span(tmp_path: Path):
    bed = tmp_path / "r.bed"
    bed.write_text("7\t1000\t1010\tX\n")
    idx = IntervalIndex.from_bed(bed)
    # deletion starting before the interval but spanning into it
    assert idx.overlaps("7", 995, "ACGTACGTAC")
    assert not idx.overlaps("7", 990, "ACGT")
    assert idx.overlaps("7", 1010, "A")
    assert not idx.overlaps("7", 1011, "A")
