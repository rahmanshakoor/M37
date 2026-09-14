"""The combining rules: ClinGen SVI points (Tavtigian 2020) as the verdict, BA1
stand-alone, PP5/BP6 never counted, and the 2015 Table 5 kept for the record."""

import pytest

from engine.agents.schema import Criterion, acmg_points, classify_points, combine_acmg, combine_richards_2015


def c(code, strength, met=True):
    return Criterion(code=code, strength=strength, met=met, justification="x", evidence_ids=["vep:1:1:A:G"])


def test_pvs1_plus_pm2_supporting_is_likely_pathogenic():
    """The commonest null-variant chain. Table 5 has no rule for it (VUS); the point
    system gives 8 + 1 = 9 → likely pathogenic — the gap the SVI points close."""
    crit = [c("PVS1", "very_strong"), c("PM2", "supporting")]
    assert acmg_points(crit) == 9 and combine_acmg(crit) == "likely_pathogenic"
    assert combine_richards_2015(crit) == "vus"


def test_pvs1_plus_moderate_is_pathogenic_by_points_likely_by_table():
    crit = [c("PVS1", "very_strong"), c("PM2", "moderate")]
    assert acmg_points(crit) == 10 and combine_acmg(crit) == "pathogenic" and combine_richards_2015(crit) == "likely_pathogenic"


def test_pvs1_plus_two_supporting_is_pathogenic():
    crit = [c("PVS1", "very_strong"), c("PM2", "supporting"), c("PP4", "supporting")]
    assert acmg_points(crit) == 10 and combine_acmg(crit) == "pathogenic" == combine_richards_2015(crit)


def test_pvs1_plus_pm3_plus_pp4_is_pathogenic():
    assert combine_acmg([c("PVS1", "very_strong"), c("PM3", "moderate"), c("PP4", "supporting")]) == "pathogenic"


def test_missense_pm2_pp3_only_is_vus():
    assert combine_acmg([c("PM2", "moderate"), c("PP3", "supporting")]) == "vus"
    assert combine_acmg([c("PM2", "supporting"), c("PP3", "supporting"), c("PM3", "supporting"), c("PP4", "supporting")]) == "vus"  # 4 points


def test_missense_pm2_pm3_pp3_pp4_is_likely_pathogenic():
    crit = [c("PM2", "moderate"), c("PM3", "moderate"), c("PP3", "supporting"), c("PP4", "supporting")]
    assert acmg_points(crit) == 6 and combine_acmg(crit) == "likely_pathogenic" == combine_richards_2015(crit)


def test_ba1_is_stand_alone_benign_whatever_else_is_claimed():
    assert combine_acmg([c("BA1", "stand_alone")]) == "benign"
    assert combine_acmg([c("BA1", "stand_alone"), c("PVS1", "very_strong"), c("PS1", "strong")]) == "benign"
    assert combine_richards_2015([c("BA1", "stand_alone"), c("PVS1", "very_strong"), c("PS1", "strong")]) == "vus"


def test_benign_side_by_points():
    assert combine_acmg([c("BS1", "strong"), c("BP4", "supporting")]) == "likely_benign"      # −5
    assert combine_acmg([c("BP1", "supporting"), c("BP4", "supporting")]) == "likely_benign"  # −2
    assert combine_acmg([c("BS1", "strong"), c("BS2", "strong")]) == "benign"                 # −8
    assert combine_acmg([c("BS1", "strong")]) == "likely_benign"                              # −4: one strong benign is likely benign by points
    assert combine_richards_2015([c("BS1", "strong")]) == "vus"                               # … and uncertain by Table 5
    # evidence on both sides nets out: 8 + 2 − 2 = 8 → likely pathogenic (Table 5 calls it a contradiction)
    both = [c("BP1", "supporting"), c("BP4", "supporting"), c("PVS1", "very_strong"), c("PM2", "moderate")]
    assert combine_acmg(both) == "likely_pathogenic" and combine_richards_2015(both) == "vus"


def test_retired_pp5_and_bp6_never_count():
    crit = [c("PVS1", "very_strong"), c("PP5", "supporting"), c("PP4", "supporting")]
    assert acmg_points(crit) == 9 and combine_acmg(crit) == "likely_pathogenic"  # not pathogenic: PP5 is 0
    assert combine_richards_2015(crit) == "vus"  # PVS1 + one supporting has no Table 5 rule
    assert combine_acmg([c("BS1", "strong"), c("BP6", "supporting")]) == "likely_benign" and acmg_points([c("BP6", "supporting")]) == 0


def test_point_categories():
    assert [classify_points(p) for p in (10, 9, 6, 5, 0, -1, -6, -7)] == \
        ["pathogenic", "likely_pathogenic", "likely_pathogenic", "vus", "vus", "likely_benign", "likely_benign", "benign"]


def test_unmet_criteria_are_ignored():
    assert combine_acmg([c("PVS1", "very_strong", met=False), c("PM2", "moderate")]) == "vus"


def test_unknown_code_rejected():
    with pytest.raises(ValueError):
        Criterion(code="PX9", strength="strong", met=True, justification="x")
