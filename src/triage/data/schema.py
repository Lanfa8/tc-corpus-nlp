"""Contrato dos dados do Medical Abstracts TC Corpus."""

import pandas as pd

CONDITION_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

LABELS: list[int] = sorted(CONDITION_NAMES)

REQUIRED_COLUMNS = ("condition_label", "medical_abstract")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame de abstracts, levantando ValueError se estiver inválido."""
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"coluna obrigatória ausente: {column}")

    unknown = set(df["condition_label"].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"rótulo desconhecido encontrado: {sorted(unknown)}")

    blank = df["medical_abstract"].isna() | (
        df["medical_abstract"].astype(str).str.strip() == ""
    )
    if blank.any():
        raise ValueError(f"{int(blank.sum())} abstract(s) vazio(s) encontrados")

    return df
