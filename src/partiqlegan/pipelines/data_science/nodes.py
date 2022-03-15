"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

import numpy as np
from typing import Dict, Tuple
from phasespace import GenParticle, nbody_decay
from phasespace.fromdecay import GenMultiDecay
import pandas as pd

def processingDummy(
    intermediate_tree: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return 