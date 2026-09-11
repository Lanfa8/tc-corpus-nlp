"""Mapa determinístico de condição clínica para nível de urgência de triagem.

Esta é uma heurística de produto para atender ao requisito de triagem por
urgência do enunciado. NÃO é um rótulo clínico validado e não substitui a
avaliação de um profissional de saúde.
"""

URGENCY_LEVELS: tuple[str, ...] = ("normal", "atencao", "urgente")

URGENCY_BY_LABEL: dict[int, str] = {
    1: "urgente",  # neoplasms
    2: "normal",  # digestive system diseases
    3: "atencao",  # nervous system diseases
    4: "urgente",  # cardiovascular diseases
    5: "normal",  # general pathological conditions
}


def urgency_for(label: int) -> str:
    """Retorna o nível de urgência de um rótulo de condição."""
    return URGENCY_BY_LABEL[int(label)]
