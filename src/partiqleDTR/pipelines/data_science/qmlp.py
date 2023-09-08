import numpy as np
import time

import torch as t
from torch import nn
import torch.nn.functional as F

import mlflow

from .utils import *
from .circuits import circuit_builder, QuantumCircuit

from .gnn_bckp import gnn

from .custom_sampler_qnn import CustomSamplerQNN

import qiskit as q
from qiskit.visualization import *
from qiskit.primitives import BackendSampler
from qiskit_machine_learning.connectors import TorchConnector

from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor

import logging

qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
log = logging.getLogger(__name__)


class QMLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob, batchnorm=True, activation=F.elu):
        super(QMLP, self).__init__()

        self.batchnorm = batchnorm
        self.evaluation_timestamp = None
        self.n_shots = n_shots
        self.padding_dropout = padding_dropout
        self.predefined_vqc = predefined_vqc
        self.predefined_iec = predefined_iec
        self.initialization_constant = initialization_constant
        self.param_rng = np.random.default_rng(seed=parameter_seed)

        self.fc1 = nn.Linear(n_in, n_hid)

        self.qc = QuantumCircuit(n_qubits)
        self.qc = circuit_builder(
            self.qc,
            self.predefined_iec,
            self.predefined_vqc,
            n_qubits,
            self.layers,
            data_reupload=self.data_reupload,
        )


        self.fc2 = nn.Linear(n_hid, n_out)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=True)
            # self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=False)  # Use this to overfit
        self.dropout_prob = do_prob
        self.activation = activation
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm_layer(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        """
        Input: (b, l, c)
        Output: (b, l, d)
        """
        # Input shape: [num_sims, num_things, num_features]
        x = self.activation(self.fc1(inputs))  # (b, l, d)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # (b, l, d)
        x = self.activation(self.fc2(x))  # (b, l, d)
        return self.batch_norm_layer(x) if self.batchnorm else x  # (b, l, d)
