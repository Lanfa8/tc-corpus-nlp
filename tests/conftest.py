import pandas as pd
import pytest

CLASS_VOCAB = {
    1: "tumor malignant carcinoma metastasis oncology growth",
    2: "gastric intestinal bowel liver hepatic digestion ulcer",
    3: "neural brain seizure cerebral cognitive neuropathy nerve",
    4: "cardiac heart arterial coronary myocardial vascular pressure",
    5: "inflammation infection fever chronic general pathology syndrome",
}


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """Corpus sintético linearmente separável, 40 exemplos por classe.

    Cada classe tem vocabulário próprio, então qualquer classificador razoável
    atinge macro-F1 alto — os testes verificam mecânica, não qualidade de modelo.
    """
    rows = []
    for label, vocab in CLASS_VOCAB.items():
        words = vocab.split()
        for i in range(40):
            rotated = words[i % len(words) :] + words[: i % len(words)]
            rows.append(
                {
                    "condition_label": label,
                    "medical_abstract": " ".join(rotated + [f"filler{i}"]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_data_dir(tmp_path, synthetic_frame):
    """Escreve o corpus sintético como os CSVs esperados pelo loader."""
    synthetic_frame.to_csv(tmp_path / "medical_tc_train.csv", index=False)
    synthetic_frame.to_csv(tmp_path / "medical_tc_test.csv", index=False)
    return tmp_path
