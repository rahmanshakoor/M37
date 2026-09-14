"""engine.sex: PAR coordinates, the tally, and what ingest records."""

from engine.sex import (FEMALE_MIN_X_HET_FRACTION, MALE_MAX_X_HET_FRACTION, MIN_X_CALLS, SexTally,
                        effective_sex, in_par)


def test_par_boundaries_grch38():
    assert in_par("X", 10_001) and in_par("X", 2_781_479) and not in_par("X", 2_781_480)
    assert in_par("X", 155_701_383) and not in_par("X", 155_701_382)
    assert in_par("Y", 56_887_903) and not in_par("Y", 20_000_000)
    assert not in_par("15", 10_001)


def test_tally_infers_male_from_low_x_heterozygosity():
    t = SexTally()
    for i in range(MIN_X_CALLS):
        t.add("X", 30_000_000 + i, "0/1" if i < MIN_X_CALLS * 0.05 else "1/1")
    t.add("X", 1_000_000, "0/1")          # PAR1: ignored
    t.add("Y", 12_000_000, "1")           # haploid Y call
    t.add("Y", 12_000_100, "0/0")         # non-carrier: ignored
    t.add("15", 40_000_000, "0/1")        # autosome: ignored
    d = t.as_dict()
    assert d["x_nonpar_carrier_calls"] == MIN_X_CALLS and d["y_nonpar_carrier_calls"] == 1
    assert d["x_nonpar_het_fraction"] <= MALE_MAX_X_HET_FRACTION and d["inferred"] == "male"


def test_tally_infers_female_and_stays_unknown_when_thin():
    t = SexTally()
    for i in range(MIN_X_CALLS):
        t.add("X", 30_000_000 + i, "0/1" if i % 2 else "1/1")
    assert t.x_het_fraction >= FEMALE_MIN_X_HET_FRACTION and t.inferred() == "female"
    thin = SexTally()
    for i in range(MIN_X_CALLS - 1):
        thin.add("X", 30_000_000 + i, "1/1")
    assert thin.inferred() == "unknown"
    assert SexTally().as_dict()["x_nonpar_het_fraction"] is None


def test_effective_sex_prefers_the_case_file():
    assert effective_sex("female", "male") == "female"
    assert effective_sex("unknown", "male") == "male"
    assert effective_sex("unknown", "unknown") == "unknown"
