"""Gera carga contra a API para popular as métricas.

Uso: python scripts/generate_load.py --requests 500 --url http://localhost:8000
"""

import argparse
import logging
import random
import time

import httpx

from triage.config.seeds import set_seed

logger = logging.getLogger(__name__)

SAMPLE_ABSTRACTS = [
    "Patient presents with acute myocardial infarction and severe chest pain "
    "radiating to the left arm, with elevated troponin levels.",
    "Histological examination revealed a malignant carcinoma with evidence of "
    "lymph node metastasis and rapid tumor growth.",
    "The patient reports recurrent epigastric pain, gastric reflux and a "
    "confirmed duodenal ulcer on endoscopy.",
    "Progressive cognitive decline with cerebral atrophy on MRI and recurrent "
    "focal seizures over the past six months.",
    "Chronic systemic inflammation with persistent low grade fever and elevated "
    "inflammatory markers of unclear etiology.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera carga contra a API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--delay", type=float, default=0.02)
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    set_seed()

    ok = errors = 0
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for index in range(args.requests):
            if random.random() < args.error_rate:
                # Payload inválido de propósito, para o painel de taxa de erro.
                response = client.post("/predict", json={"text": ""})
            else:
                response = client.post(
                    "/predict", json={"text": random.choice(SAMPLE_ABSTRACTS)}
                )
            if response.status_code == 200:
                ok += 1
            else:
                errors += 1
            if index % 50 == 0:
                logger.info("%d/%d requisições", index, args.requests)
            time.sleep(args.delay)

    logger.info("concluído: %d ok, %d erros", ok, errors)


if __name__ == "__main__":
    main()
