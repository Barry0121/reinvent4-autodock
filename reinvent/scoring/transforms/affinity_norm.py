"""Transform affinity prediction to normalized score"""

__all__ = ["AffinityNormalization"]
from dataclasses import dataclass

import numpy as np
from .transform import Transform


@dataclass
class Parameters:
    type: str
    worst_affinity: float # usually around 0
    best_affinity: float # very negative


class AffinityNormalization(Transform, param_cls=Parameters):
    def __init__(self, params: Parameters):
        super().__init__(params)

        self.worst_affinity = params.worst_affinity
        self.best_affinity = params.best_affinity

    def __call__(self, values) -> np.ndarray:
        values = np.array(values, dtype=np.float32)
        # clip to the specified range
        clipped = np.clip(values, self.best_affinity, self.worst_affinity)
        transformed = (clipped - self.worst_affinity) / (self.best_affinity - self.worst_affinity)

        return transformed
