"""Exporta o pipeline sklearn treinado (TF-IDF + classificador) para ONNX.

O pipeline inteiro vira um grafo ONNX — a vetorização TF-IDF inclusive — então
a inferência sai completamente do Python.

Uso: python scripts/export_onnx.py [--models-dir models]

Três correções são necessárias para que a paridade sklearn↔ONNX fique dentro
de 1e-4 no pipeline real (`max_features=50_000`, corpus completo), não apenas
em amostras pequenas de vocabulário reduzido:

1. `options={id(tfidf): {"locale": "C"}}` — o `StringNormalizer` gerado pelo
   conversor do vetorizador usa por padrão o locale do sistema operacional
   (`en_US.UTF-8`), ausente neste ambiente e na imagem `python:3.10-slim`
   usada em produção; sem essa opção a sessão do ONNX Runtime falha ao
   inicializar. O locale "C" não depende de locale nenhum instalado.

2. `register_traceable_text()` (de `skl2onnx.sklapi.sklearn_text_onnx`)
   registra os conversores para `TraceableTfidfVectorizer`, usada em
   `triage.features.text.build_vectorizer` no lugar do `TfidfVectorizer`
   comum — ver a docstring de `build_vectorizer` para o porquê (decomposição
   exata de n-gramas vs. adivinhação por substring).

3. `_register_sublinear_tf_fix`: o conversor `SklearnTfidfTransformer` do
   skl2onnx 1.20.0 implementa o termo sublinear como ``log(1 + tf)``, mas o
   scikit-learn usa ``1 + log(tf)`` (só para tf > 0; termos ausentes
   permanecem 0). As duas fórmulas só coincidem quando toda palavra de um
   documento aparece exatamente uma vez — em texto real, com palavras
   repetidas, elas divergem e a normalização L2 não cancela a diferença.
   Esta função substitui esse conversor por uma versão que reproduz a
   fórmula exata do scikit-learn, mascarando os termos com tf=0 para não
   aplicar log(0).

4. `options={id(clf): {"zipmap": False}}` — faz a saída de probabilidades do
   classificador ser um tensor `(n, 5)` em vez de uma lista de dicionários,
   mais rápido e mais simples de consumir. Com essa opção o grafo produz
   duas saídas (`label`, `probabilities`); `OnnxPredictor` lê
   `get_outputs()[1]`.

Medido em 500 documentos reais do conjunto de teste com as quatro correções:
diferença absoluta máxima de probabilidade ~1.9e-7, 100% de concordância de
classe com o backend sklearn. Sem as correções 2 e 3 (mesmo já sem
`stop_words` e com `token_pattern`), a diferença chega a ~0.12 e a
concordância de classe cai para ~96%.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType

from triage.config.settings import get_settings
from triage.models.artifacts import load_pipeline

logger = logging.getLogger(__name__)

ONNX_FILE = "pipeline.onnx"


def _register_sublinear_tf_fix() -> None:
    """Corrige o conversor de `TfidfTransformer` para `sublinear_tf=True`.

    Substitui `log(1 + tf)` (o que o skl2onnx 1.20.0 gera) por
    `tf > 0 ? 1 + log(tf) : 0` (o que o scikit-learn realmente calcula).
    Ver docstring do módulo para a evidência da divergência.
    """
    from skl2onnx import update_registered_converter
    from skl2onnx.common._apply_operation import (
        apply_add,
        apply_identity,
        apply_log,
        apply_mul,
        apply_normalizer,
    )
    from skl2onnx.common.data_types import guess_numpy_type, guess_proto_type
    from skl2onnx.proto import onnx_proto
    from skl2onnx.shape_calculators.tfidf_transformer import (
        calculate_sklearn_tfidf_transformer_output_shapes,
    )
    from sklearn.feature_extraction.text import TfidfTransformer

    def convert_sklearn_tfidf_transformer_fixed(scope, operator, container) -> None:
        dtype = guess_numpy_type(operator.inputs[0].type)
        float_type = dtype if dtype == np.float64 else np.float32
        proto_dtype = guess_proto_type(operator.inputs[0].type)
        if proto_dtype != onnx_proto.TensorProto.DOUBLE:
            proto_dtype = onnx_proto.TensorProto.FLOAT
        op = operator.raw_operator
        data = operator.input_full_names
        output_name = scope.get_unique_variable_name("tfidftr_output")

        if op.sublinear_tf:
            logged = scope.get_unique_variable_name("logged")
            apply_log(scope, data, logged, container)
            plus1 = scope.get_unique_variable_name("plus1")
            ones_name = scope.get_unique_variable_name("ones_sub")
            n_columns = operator.inputs[0].type.shape[1]
            ones = np.ones((n_columns,), dtype=float_type)
            container.add_initializer(ones_name, proto_dtype, [n_columns], ones)
            apply_add(scope, [logged, ones_name], plus1, container, broadcast=1)

            zero_name = scope.get_unique_variable_name("zero_sub")
            container.add_initializer(zero_name, proto_dtype, [1], [0])
            mask_name = scope.get_unique_variable_name("mask_sub")
            container.add_node(
                "Greater",
                [*data, zero_name],
                mask_name,
                name=scope.get_unique_operator_name("Greater"),
            )
            masked = scope.get_unique_variable_name("masked_sub")
            container.add_node(
                "Where",
                [mask_name, plus1, zero_name],
                masked,
                name=scope.get_unique_operator_name("Where"),
            )
            data = [masked]

        if op.use_idf:
            idf = op.idf_.astype(float_type).ravel()
            idfcst = scope.get_unique_variable_name("idfcst")
            container.add_initializer(idfcst, proto_dtype, [len(idf)], idf)
            apply_mul(scope, [*data, idfcst], output_name, container, broadcast=1)
        else:
            output_name = data[0]

        if op.norm is not None:
            norm_name = scope.get_unique_variable_name("tfidftr_norm")
            apply_normalizer(
                scope,
                output_name,
                norm_name,
                container,
                norm=op.norm.upper(),
                use_float=float_type == np.float32,
            )
            output_name = norm_name

        apply_identity(scope, output_name, operator.output_full_names, container)

    update_registered_converter(
        TfidfTransformer,
        "SklearnTfidfTransformer",
        calculate_sklearn_tfidf_transformer_output_shapes,
        convert_sklearn_tfidf_transformer_fixed,
        overwrite=True,
        options={"nan": [True, False]},
    )


def export(models_dir: Path) -> Path:
    """Converte o pipeline salvo em `models_dir` para ONNX."""
    from skl2onnx.sklapi.sklearn_text_onnx import register as register_traceable_text

    register_traceable_text()
    _register_sublinear_tf_fix()
    pipeline, _ = load_pipeline(models_dir)

    options = {
        id(pipeline.named_steps["tfidf"]): {"locale": "C"},
        id(pipeline.named_steps["clf"]): {"zipmap": False},
    }
    onnx_model = to_onnx(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        target_opset=15,
        options=options,
    )

    destination = Path(models_dir) / ONNX_FILE
    destination.write_bytes(onnx_model.SerializeToString())

    joblib_size = (Path(models_dir) / "pipeline.joblib").stat().st_size
    onnx_size = destination.stat().st_size
    logger.info(
        "exportado %s (%.2f MB); joblib tem %.2f MB",
        destination,
        onnx_size / 1e6,
        joblib_size / 1e6,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta o pipeline para ONNX")
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    models_dir = args.models_dir or get_settings().models_dir
    export(models_dir)


if __name__ == "__main__":
    main()
