"""Carga e limpeza dos CSVs do corpus."""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from triage.config.seeds import SEED
from triage.config.settings import get_settings
from triage.data.schema import validate

logger = logging.getLogger(__name__)

SPLIT_FILES: dict[str, str] = {
    "train": "medical_tc_train.csv",
    "test": "medical_tc_test.csv",
}


def load_raw(path: Path) -> pd.DataFrame:
    """Lê um CSV do corpus sem transformação."""
    logger.info("lendo %s", path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove abstracts vazios e duplicados, normaliza tipos e espaços."""
    out = df.copy()
    out["medical_abstract"] = out["medical_abstract"].astype("string").str.strip()
    out = out[out["medical_abstract"].notna() & (out["medical_abstract"] != "")]
    out["medical_abstract"] = out["medical_abstract"].astype(str)
    out["condition_label"] = out["condition_label"].astype(int)
    before = len(out)
    out = out.drop_duplicates(subset=["medical_abstract"])
    if before != len(out):
        logger.info("removidos %d abstracts duplicados", before - len(out))
    return out.reset_index(drop=True)


def load_split(
    split: Literal["train", "test"], data_dir: Path | None = None
) -> pd.DataFrame:
    """Carrega, limpa e valida um split do corpus."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split inválido: {split!r}; use 'train' ou 'test'")
    directory = data_dir if data_dir is not None else get_settings().data_dir
    return validate(clean(load_raw(Path(directory) / SPLIT_FILES[split])))


def train_val_split(
    df: pd.DataFrame, val_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o treino em treino/validação de forma estratificada e reprodutível."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["condition_label"],
    )
    logger.info("split: %d treino / %d validação", len(train_df), len(val_df))
    return train_df, val_df
