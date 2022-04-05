"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""
import numpy as np
from typing import Dict, Tuple

import torch
import torchvision
import torchdata as td

data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

class DecayTreeDataset(torch.utils.data.dataset):
    def __init__(self, dataset:Dict[str, np.ndarray]):

        self.decayTreeDataset = dataset
        
        raise NotImplementedError("Sorry, not yet..")

    def __len__(self):
        return len(self.decayTreeDataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        raise NotImplementedError("Sorry, not yet..")
        
def train_test_split(
    intermediate_tree: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:



    return 