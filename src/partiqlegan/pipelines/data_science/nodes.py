"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

import numpy as np
from typing import Dict, Tuple, List
from phasespace import GenParticle, nbody_decay
from phasespace.fromdecay import GenMultiDecay
import pandas as pd

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNNfrom qiskit_machine_learning.connectors import TorchConnector

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F


class Generator(TorchConnector):
    def __init__(self,
                    latentDim:int=100, 
                    outputShape:Tuple(int,int,int)=(1,4,4)
    ) -> None:
        num_inputs = outputShape[1]*outputShape[2]
        feature_map = ZZFeatureMap(num_inputs)
        ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

        # Define quantum circuit of num_qubits = input dim
        # Append feature map and ansatz
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))


        # Define CircuitQNN and initial setup
        parity = lambda x: "{:b}".format(x).count("1") % 2  # optional interpret function
        output_shape = 2  # parity = 0, 1
        qnn2 = CircuitQNN(
            qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=output_shape,
            quantum_instance=qi,
        )

# Set up PyTorch module
# Reminder: If we don't explicitly declare the initial weights
# they are chosen uniformly at random from [-1, 1].
# initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn2.num_weights) - 1)

    #     super(Generator, self).__init__(self.model, initialWeights)

    # def forward(self, z):
    #     graph = self.model(z)
    #     graph = graph.view(graph.size(0), *self.outputShape)
    #     return graph

class Discriminator(nn.Module):
    def __init__(self, 
                    inputShape:Tuple(int,int,int)=(1,4,4)
    ) -> None:
        super(Discriminator, self).__init__()

        self.inputShape = inputShape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.inputShape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

def train_discriminator(
    generator_input: Dict
) -> nn.Module:
    pass

def train_generator(
    generator_input: Tuple[List, List]
) -> nn.Module:
    pass