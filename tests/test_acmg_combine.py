import pytest

from engine.agents.schema import Criterion, combine_acmg


def c(code, strength, met=True):
    return Criterion(code=code, strength=strength, met=met, justification="x", evidence_ids=["vep:1:1:A:G"])


def test_pvs1_plus_pm2_supporting_is_likely_pathogenic_not_pathogenic():
    # PVS1 + one supporting → not enough for P; PVS1 alone + 1 PP is not in Table 5 for LP either → VUS
    assert combine_acmg([c("PVS1", "very_strong"), c("PM2", "supporting")]) == "vus"


def test_pvs1_plus_moderate_is_likely_pathogenic():
    assert combine_acmg([c("PVS1", "very_strong"), c("PM2", "moderate")]) == "likely_pathogenic"


def test_pvs1_plus_two_supporting_is_pathogenic():
    assert combine_acmg([c("PVS1", "very_strong"), c("PM2", "supporting"), c("PP4", "supporting")]) == "pathogenic"


def test_pvs1_plus_pm3_plus_pp4_is_pathogenic():
    # nonsense + in trans with pathogenic + phenotype specificity: PVS1 + PM + PP → pathogenic
    assert combine_acmg([c("PVS1", "very_strong"), c("PM3", "moderate"), c("PP4", "supporting")]) == "pathogenic"


def test_missense_pm2_pp3_only_is_vus():
    assert combine_acmg([c("PM2", "moderate"), c("PP3", "supporting")]) == "vus"


def test_missense_pm2_pm3_pp3_pp4_is_likely_pathogenic():
    assert combine_acmg([c("PM2", "moderate"), c("PM3", "moderate"), c("PP3", "supporting"), c("PP4", "supporting")]) == "likely_pathogenic"


def test_ba1_is_benign_and_contradiction_is_vus():
    assert combine_acmg([c("BA1", "stand_alone")]) == "benign"
    assert combine_acmg([c("BA1", "stand_alone"), c("PVS1", "very_strong"), c("PS1", "strong")]) == "vus"


def test_benign_combinations_bs1_plus_bp_or_two_bp_is_likely_benign_two_bs_is_benign():
    # Table 5 benign side: BS1 + one supporting, or two supporting → likely benign; two strong → benign
    assert combine_acmg([c("BS1", "strong"), c("BP4", "supporting")]) == "likely_benign"
    assert combine_acmg([c("BP1", "supporting"), c("BP4", "supporting")]) == "likely_benign"
    assert combine_acmg([c("BS1", "strong"), c("BS2", "strong")]) == "benign"
    assert combine_acmg([c("BS1", "strong")]) == "vus"  # one strong benign alone is not enough
    # likely benign against likely pathogenic is a contradiction → VUS, as on the pathogenic side
    assert combine_acmg([c("BP1", "supporting"), c("BP4", "supporting"), c("PVS1", "very_strong"), c("PM2", "moderate")]) == "vus"


def test_unmet_criteria_are_ignored():
    assert combine_acmg([c("PVS1", "very_strong", met=False), c("PM2", "moderate")]) == "vus"


def test_unknown_code_rejected():
    with pytest.raises(ValueError):
        Criterion(code="PX9", strength="strong", met=True, justification="x")
