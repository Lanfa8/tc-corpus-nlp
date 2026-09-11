import pandas as pd
import pytest

from triage.data.loader import clean, load_split, train_val_split


@pytest.fixture
def synthetic_csv(tmp_path):
    """Gera um CSV sintético com as 5 classes, suficiente para split estratificado."""
    rows = []
    for label in range(1, 6):
        for i in range(20):
            rows.append(
                {
                    "condition_label": label,
                    "medical_abstract": f"abstract sobre condicao {label} numero {i}",
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "medical_tc_train.csv"
    df.to_csv(path, index=False)
    return tmp_path


def test_clean_drops_null_and_blank_abstracts():
    df = pd.DataFrame(
        {
            "condition_label": [1, 2, 3],
            "medical_abstract": ["texto valido", None, "   "],
        }
    )
    result = clean(df)
    assert len(result) == 1
    assert result.iloc[0]["medical_abstract"] == "texto valido"


def test_clean_strips_whitespace_and_casts_label():
    df = pd.DataFrame(
        {"condition_label": ["1"], "medical_abstract": ["  texto com espaco  "]}
    )
    result = clean(df)
    assert result.iloc[0]["medical_abstract"] == "texto com espaco"
    assert result.iloc[0]["condition_label"] == 1


def test_clean_drops_duplicates():
    df = pd.DataFrame(
        {"condition_label": [1, 1], "medical_abstract": ["mesmo texto", "mesmo texto"]}
    )
    assert len(clean(df)) == 1


def test_load_split_reads_and_validates(synthetic_csv):
    df = load_split("train", data_dir=synthetic_csv)
    assert len(df) == 100
    assert set(df.columns) == {"condition_label", "medical_abstract"}


def test_load_split_rejects_unknown_split(synthetic_csv):
    with pytest.raises(ValueError, match="split"):
        load_split("validation", data_dir=synthetic_csv)


def test_train_val_split_is_stratified_and_reproducible(synthetic_csv):
    df = load_split("train", data_dir=synthetic_csv)
    train_a, val_a = train_val_split(df, val_size=0.2)
    train_b, val_b = train_val_split(df, val_size=0.2)

    assert len(val_a) == 20
    assert len(train_a) == 80
    # Estratificado: 4 exemplos de cada classe na validação
    assert val_a["condition_label"].value_counts().to_dict() == dict.fromkeys(
        range(1, 6), 4
    )
    # Reprodutível com a mesma seed
    assert train_a.index.tolist() == train_b.index.tolist()
    assert val_a.index.tolist() == val_b.index.tolist()
