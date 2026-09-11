import pytest

from triage.data.schema import LABELS
from triage.urgency import URGENCY_BY_LABEL, URGENCY_LEVELS, urgency_for


def test_every_label_has_an_urgency():
    assert set(URGENCY_BY_LABEL) == set(LABELS)


def test_all_urgencies_are_valid_levels():
    assert set(URGENCY_BY_LABEL.values()) <= set(URGENCY_LEVELS)


def test_mapping_matches_the_spec():
    assert urgency_for(1) == "urgente"  # neoplasms
    assert urgency_for(4) == "urgente"  # cardiovascular diseases
    assert urgency_for(3) == "atencao"  # nervous system diseases
    assert urgency_for(2) == "normal"  # digestive system diseases
    assert urgency_for(5) == "normal"  # general pathological conditions


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        urgency_for(99)
