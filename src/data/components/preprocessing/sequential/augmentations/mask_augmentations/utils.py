from typing import Dict

import numpy as np


def get_sequence_length(sequential_features: Dict[str, np.ndarray]) -> int:
    return next(sequential_features.values().__iter__()).shape[0]
