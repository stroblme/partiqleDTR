"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple
import torchdata

def train_test_split(
    intermediate_tree: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:



    return 