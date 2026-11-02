from __future__ import annotations
import itertools
import math
from typing import List

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def mean_pairwise_cosine(embeddings: List[List[float]]) -> float:
    r = len(embeddings)
    if r < 2:
        return 1.0
    sims = [
        cosine_similarity(embeddings[i], embeddings[j])
        for i, j in itertools.combinations(range(r), 2)
    ]
    return sum(sims) / len(sims)
