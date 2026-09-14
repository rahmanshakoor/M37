import pytest

from engine.contigs import canonical, is_primary, naming_style, primary_names, to_annotation, to_submission


@pytest.mark.parametrize("raw,expected", [
    ("15", "15"), ("chr15", "15"), ("CHR15", "15"),
    ("X", "X"), ("chrX", "X"), ("y", "Y"),
    ("M", "MT"), ("MT", "MT"), ("chrM", "MT"), ("chrMT", "MT"),
    (" 1 ", "1"),
])
def test_canonical_primary(raw, expected):
    assert canonical(raw) == expected


@pytest.mark.parametrize("raw", [
    "1_KI270706v1_random", "chr1_KI270706v1_random", "chrUn_KI270302v1",
    "HLA-A*01:01:01:01", "chr19_KI270938v1_alt", "hs38d1_decoy", "23", "0", "",
])
def test_canonical_non_primary(raw):
    assert canonical(raw) is None
    assert not is_primary(raw)


def test_submission_form_is_what_the_scorer_compares():
    # The silent-zero trap: the VCF says "15", the scorer wants "chr15".
    assert to_submission("15") == "chr15"
    assert to_submission("chr15") == "chr15"
    assert to_submission("M") == "chrM"
    assert to_submission("MT") == "chrM"
    assert to_submission("X") == "chrX"


def test_annotation_form_is_ensembl():
    assert to_annotation("chr15") == "15"
    assert to_annotation("chrM") == "MT"
    assert to_annotation("M") == "MT"


def test_non_primary_cannot_be_converted():
    with pytest.raises(ValueError):
        to_submission("1_KI270706v1_random")
    with pytest.raises(ValueError):
        to_annotation("chrUn_KI270302v1")


def test_naming_style_detection():
    assert naming_style(["1", "2", "X", "M", "1_KI270706v1_random"]) == "ensembl"
    assert naming_style(["chr1", "chr2", "chrX", "chrM", "chrUn_KI270302v1"]) == "ucsc"
    with pytest.raises(ValueError):
        naming_style(["foo", "bar"])


def test_primary_names_keep_file_spelling_and_order():
    header = ["1", "2", "M", "1_KI270706v1_random", "X"]
    assert primary_names(header) == ["1", "2", "M", "X"]
    assert primary_names(["chr3", "chrM", "chrUn_x"]) == ["chr3", "chrM"]
