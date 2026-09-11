"""Normalização de texto e construção do vetorizador TF-IDF."""

import re

from skl2onnx.sklapi import TraceableTfidfVectorizer

_NON_LETTER = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Minúsculas, remove dígitos e pontuação, colapsa espaços.

    Não é mais usada como preprocessor do vetorizador (ver `build_vectorizer`) —
    um preprocessor Python arbitrário não pode ser exportado para ONNX
    (`skl2onnx` recusa com `NotImplementedError: Custom preprocessor cannot be
    converted into ONNX.`). Serve agora de oráculo de referência: tokenizar o
    texto bruto com `token_pattern=r"[a-z]{2,}"` e `lowercase=True` produz
    exatamente os mesmos tokens que tokenizar a saída desta função sobre o
    mesmo texto — ver `test_vectorizer_analyzer_matches_normalize_text_oracle`
    em `tests/features/test_text.py`, que prova a equivalência.
    """
    lowered = str(text).lower()
    letters_only = _NON_LETTER.sub(" ", lowered)
    return _WHITESPACE.sub(" ", letters_only).strip()


def build_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.9,
    max_features: int = 50_000,
) -> TraceableTfidfVectorizer:
    """Cria o TfidfVectorizer usado pelo pipeline de treino e de inferência.

    Duas mudanças em relação à versão pré-ONNX, ambas necessárias para que a
    conversão sklearn -> ONNX preserve o comportamento do modelo:

    1. `token_pattern=r"[a-z]{2,}"` no lugar do `preprocessor`/`stop_words`
       customizados. `skl2onnx` não converte um `preprocessor` Python
       arbitrário. `stop_words="english"` combinado com bigramas também não é
       seguro de converter: o scikit-learn remove stop words antes de montar
       os n-gramas, e o conversor ONNX não reproduz essa ordem — medido em
       texto real, a diferença de probabilidade sklearn↔ONNX chega a 2.2e-2
       nessa combinação. Sem `stop_words`, a divergência cai para ~1.6e-07 em
       amostras pequenas.

    2. `TraceableTfidfVectorizer` (subclasse de `TfidfVectorizer` do próprio
       skl2onnx — mesmo `fit`/`transform`/`vocabulary_`) no lugar do
       `TfidfVectorizer` padrão. Isso importa na escala real de produção
       (`max_features=50_000`, `max_df=0.9`, corpus completo): o conversor do
       skl2onnx para o `TfidfVectorizer` comum não guarda a decomposição de
       cada n-grama em tokens: quando `max_df` descarta uma palavra muito
       frequente ("the", "of") como unigrama mas ela sobrevive dentro de um
       bigrama ("of the"), o conversor precisa *adivinhar* a decomposição
       comparando substrings contra o vocabulário de unigramas — e, sem "the"
       nesse vocabulário, o bigrama vira um token que o grafo ONNX não
       consegue formar em tempo de inferência, zerando essa feature. Medido
       no vocabulário real treinado (50 000 termos): 31.9% dos 37 527
       bigramas têm ao menos um componente ausente do vocabulário de
       unigramas, e a paridade sklearn↔ONNX com o `TfidfVectorizer` comum
       chega a ~0.12 de diferença absoluta de probabilidade — bem acima da
       tolerância de 1e-4, mesmo já sem `stop_words`. `TraceableTfidfVectorizer`
       guarda a tupla de palavras de cada n-grama no fit e não precisa
       adivinhar; com ela, a mesma medição cai para ~2e-7 (ver
       `scripts/export_onnx.py` para a segunda correção necessária, no
       conversor de `sublinear_tf`, também exigida apenas na escala real).
    """
    return TraceableTfidfVectorizer(
        lowercase=True,
        token_pattern=r"[a-z]{2,}",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
