import pandas as pd
import pytest

from triage.data.schema import CONDITION_NAMES, LABELS, validate


def test_five_conditions_are_mapped():
    assert LABELS == [1, 2, 3, 4, 5]
    assert CONDITION_NAMES[1] == "neoplasms"
    assert CONDITION_NAMES[2] == "digestive system diseases"
    assert CONDITION_NAMES[3] == "nervous system diseases"
    assert CONDITION_NAMES[4] == "cardiovascular diseases"
    assert CONDITION_NAMES[5] == "general pathological conditions"


def test_validate_accepts_well_formed_frame():
    df = pd.DataFrame(
        {"condition_label": [1, 3], "medical_abstract": ["texto a", "texto b"]}
    )
    assert validate(df).equals(df)


def test_validate_rejects_missing_column():
    df = pd.DataFrame({"condition_label": [1]})
    with pytest.raises(ValueError, match="medical_abstract"):
        validate(df)


def test_validate_rejects_unknown_label():
    df = pd.DataFrame({"condition_label": [9], "medical_abstract": ["texto"]})
    with pytest.raises(ValueError, match="rótulo"):
        validate(df)


def test_validate_rejects_empty_abstract():
    df = pd.DataFrame({"condition_label": [1], "medical_abstract": ["   "]})
    with pytest.raises(ValueError, match="vazio"):
        validate(df)
