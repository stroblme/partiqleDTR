import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
import time

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

import mlflow

from .utils import *

import qiskit as q
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector

import logging

qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
log = logging.getLogger(__name__)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(
        self,
        n_in,
        n_hid,
        n_out,
        do_prob,
        batchnorm=True,
        activation=F.elu,
        device=t.device("cpu"),
    ):
        super(MLP, self).__init__()

        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(n_in, n_hid, device=device)
        self.fc2 = nn.Linear(n_hid, n_out, device=device)
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


class sqgnn(nn.Module):
    """NRI model built off the official implementation.

    Contains adaptations to make it work with our use case, plus options for extra layers to give it some more oomph

    Args:
        infeatures (int): Number of input features
        num_classes (int): Number of classes in ouput prediction
        nblocks (int): Number of NRI blocks in the model
        dim_feedforward (int): Width of feedforward layers
        initial_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) before NRI blocks
        block_additional_mlp_layers (int): Number of additional MLP (2 feedforward, 1 batchnorm (optional)) within NRI blocks, when 0 the total number is one.
        final_mlp_layers (int): Number of MLP (2 feedforward, 1 batchnorm (optional)) after NRI blocks
        dropout (float): Dropout rate
        factor (bool): Whether to use NRI blocks at all (useful for benchmarking)
        tokenize ({int: int}): Dictionary of tokenized features to embed {index_of_feature: num_tokens}
        embedding_dims (int): Number of embedding dimensions to use for tokenized features
        batchnorm (bool): Whether to use batchnorm in MLP layers
    """

    def __init__(
        self,
        n_momenta,  # d
        n_classes,  # l
        n_blocks=3,
        dim_feedforward=128,  # ff
        n_layers_mlp=2,
        n_additional_mlp_layers=2,
        n_final_mlp_layers=2,
        skip_block=False,
        skip_global=False,
        dropout_rate=0.3,
        factor=True,
        tokenize=-1,
        embedding_dims=-1,
        batchnorm=True,
        symmetrize=True,
        pre_trained_model=None,
        n_fsps: int = -1,
        device: str = "cpu",
        # data_reupload = True,
        **kwargs,
    ):
        super(sqgnn, self).__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"
        # n_fsps = 4
        self.layers = n_fsps  # dim_feedforward//8
        self.num_classes = n_classes
        self.factor = factor
        self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = n_additional_mlp_layers
        self.skip_block = skip_block
        self.skip_global = skip_global
        # self.max_leaves = max_leaves

        # self.initial_mlp = pre_trained_module.initial_mlp
        # self.blocks = pre_trained_module.blocks
        # self.final_mlp = pre_trained_module.final_mlp

        self.qi = q.utils.QuantumInstance(
            q.Aer.get_backend("aer_simulator_statevector")
        )

        self.enc_params = []
        self.var_params = []

        self.device = t.device(
            "cuda" if t.cuda.is_available() and device != "cpu" else "cpu"
        )

        def gen_encoding_params(n_qubits, identifier):
            q_params = []
            for i in range(n_qubits):
                q_params.append([])
                for j in range(n_momenta):
                    q_params[i].append(q.circuit.Parameter(f"{identifier}_{i}_{j}"))
            return q_params

        def encoding(qc, n_qubits, q_params, identifier):
            for i in range(n_qubits):
                energy = q_params[i][0]

                px = (
                    q_params[i][1] * energy * t.pi,
                    i,
                    f"{identifier[:-3]}_rx_{i}",
                )
                qc.rx(*px)
                py = (
                    q_params[i][2] * energy * t.pi,
                    i,
                    f"{identifier[:-3]}_ry_{i}",
                )
                qc.ry(*py)
                pz = (
                    q_params[i][3] * energy * t.pi,
                    i,
                )  # rz does not accept identifier
                qc.rz(*pz)
                # qc.ry(*param)

        def variational(qc, n_qubits, identifier):
            for i in range(n_qubits):
                if i == 0:
                    qc.crx(
                        q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_crx_{i+1}_{i}",
                    )
                    qc.cry(
                        q.circuit.Parameter(f"{identifier}_cry_{i+1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_cry_{i+1}_{i}",
                    )
                    qc.crz(
                        q.circuit.Parameter(f"{identifier}_crz_{i+1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_crz_{i+1}_{i}",
                    )
                else:
                    qc.crx(
                        q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_crx_{i+1}_{i}",
                    )
                    qc.cry(
                        q.circuit.Parameter(f"{identifier}_cry_{i+1}_{i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_cry_{i+1}_{i}",
                    )
                    qc.crz(
                        q.circuit.Parameter(f"{identifier}_crz_{i+1}_{i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_crz_{i+1}_{i}",
                    )

        def circuit_builder(qc, n_qubits, n_hidden):
            enc_params = gen_encoding_params(n_qubits, f"enc")
            for i in range(n_hidden):
                encoding(qc, n_qubits, enc_params, f"enc_{i}")
                qc.barrier()
                variational(qc, n_qubits, f"var_{i}")
                qc.barrier()

        log.info(
            f"Building Quantum Circuit with {self.layers} layers and {n_classes} qubits"
        )
        self.qc = q.QuantumCircuit(n_fsps)
        circuit_builder(self.qc, n_fsps, self.layers)

        mlflow.log_figure(self.qc.draw(output="mpl"), f"circuit.png")

        for param in self.qc.parameters:
            if "enc" in param.name:
                self.enc_params.append(param)
            else:
                self.var_params.append(param)
        log.info(
            f"Encoding Parameters: {len(self.enc_params)}, Variational Parameters: {len(self.var_params)}"
        )

        def interpreter(x):
            print(f"Interpreter Input {x}")
            return x

        start = time.time()
        self.qnn = CircuitQNN(
            self.qc,
            self.enc_params,
            self.var_params,
            quantum_instance=self.qi,
            # interpret=interpreter,
            # output_shape=(n_fsps**2, n_classes),
            # sampling=True,
            input_gradients=True,
        )
        self.initial_weights = 0.1 * (
            2 * q.utils.algorithm_globals.random.random(self.qnn.num_weights) - 1
        )
        log.info(f"Transpilation took {time.time() - start}")
        self.quantum_layer = TorchConnector(
            self.qnn, initial_weights=self.initial_weights
        )
        log.info(f"Initialization done")

        self.fc1_out = MLP(
            self.num_classes,
            self.num_classes,
            self.num_classes,
            dropout_rate,
            batchnorm,
            activation=F.elu,
            device=self.device,
        )
        self.fc2_out = MLP(
            self.num_classes,
            2 * self.num_classes,
            self.num_classes,
            dropout_rate,
            batchnorm,
            activation=F.elu,
            device=self.device,
        )
        self.fc3_out = MLP(
            2 * self.num_classes,
            2 * self.num_classes,
            self.num_classes,
            dropout_rate,
            batchnorm,
            activation=F.elu,
            device=self.device,
        )
        # self.fc3_out = MLP(self.num_classes, self.num_classes, self.num_classes, dropout_rate, batchnorm)

        # self.fc_out = nn.Linear(dim_feedforward, self.num_classes)

    def edge2node(self, x, rel_rec):
        """
        Input: (b, l*l, d), (b, l*l, l)
        Output: (b, l, d)
        """
        # TODO assumes that batched matrix product just works
        # TODO these do not have to be members
        incoming = t.matmul(rel_rec.permute(0, 2, 1), x)  # (b, l, d)
        denom = rel_rec.sum(1)[:, 1]
        return incoming / denom.reshape(-1, 1, 1)  # (b, l, d)
        # return incoming / incoming.size(1)  # (b, l, d)

    def node2edge(self, x, rel_rec, rel_send):
        """
        Input: (b, l, d), (b, l*(l-1), l), (b, l*(l-1), l)
        Output: (b, l*l(l-1), 2d)
        """
        # TODO assumes that batched matrix product just works
        receivers = t.matmul(rel_rec, x)  # (b, l*l, d)
        senders = t.matmul(rel_send, x)  # (b, l*l, d)
        edges = t.cat([senders, receivers], dim=2)  # (b, l*l, 2d)

        return edges

    def forward(self, inputs):
        """
        Input: (l, b, d)
        Output: (b, c, l, l)
        """
        if isinstance(inputs, (list, tuple)):
            inputs, rel_rec, rel_send = inputs
        else:
            rel_rec = None
            rel_send = None

        n_leaves, batch, feats = inputs.size()
        assert inputs.device == self.device

        # NOTE create rel matrices on the fly if not given as input
        if rel_rec is None:
            # rel_rec = t.eye(
            #     n_leaves,
            #     device=device
            # ).repeat_interleave(n_leaves-1, dim=1).T  # (l*(l-1), l)
            # rel_rec = rel_rec.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_rec = construct_rel_recvs([inputs.size(0)], device=self.device)

        if rel_send is None:
            # rel_send = t.eye(n_leaves, device=device).repeat(n_leaves, 1)
            # rel_send[t.arange(0, n_leaves*n_leaves, n_leaves + 1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*(l-1), l)
            # rel_send = rel_send.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_send = construct_rel_sends([inputs.size(0)], device=self.device)

        x = inputs.permute(1, 0, 2)  # (b, l, m)

        # x = x.flatten()
        x = x.reshape(batch, n_leaves*feats)
        x = t.nn.functional.pad(x, (0, feats**2 - x.shape[1]), mode="constant", value=0)

        x = self.quantum_layer(x)

        def build_binary_permutation_indices(digits):
            n_permutations = (digits**2 - digits) // 2
            permutations_indices = []
            for i in range(2**digits):
                if bin(i).count("1") == 2:
                    permutations_indices.append(i)
            assert len(permutations_indices) == n_permutations
            return permutations_indices

        def get_binary_shots(result, permutations_indices, out_shape):
            lcag = t.zeros(out_shape)
            lcag = lcag.to(result.device)
            for i in range(out_shape[0]):
                lcag[i][np.tril_indices_from(lcag[0], k=-1)] = result[i, permutations_indices]
            # lcag[:, np.tril_indices_from(lcag[:], k=-1)] = result[:, permutations_indices]
            lcag = lcag + t.transpose(lcag, 1, 2)
            # lcag = lcag - t.diag(t.ones(out_shape[0]).to(result.device))
            return lcag

        # x = x.reshape(batch, x.shape[0], 1)
        # x = self.fc_out(x)

        x = get_binary_shots(
            x, build_binary_permutation_indices(n_leaves), (batch, n_leaves, n_leaves)
        )

        # b = t.tensor([[i] for i in range(self.num_classes)]).repeat(1,n_leaves**2).reshape(batch,self.num_classes,n_leaves*n_leaves)
        # b = b.to(x.device)

        # ea = t.eye(x.shape[0]).repeat(1,self.num_classes).transpose(0,1).reshape(batch,self.num_classes,n_leaves*n_leaves)
        # ed = t.ones(x.shape).repeat(1,self.num_classes).transpose(0,1).reshape(batch,self.num_classes,n_leaves*n_leaves) - ea
        # ea = ea.to(x.device)
        # ed = ed.to(x.device)

        # x = (
        #     x.repeat(1, self.num_classes)
        #     .transpose(0, 1)
        #     .reshape(batch, self.num_classes, n_leaves * n_leaves)
        # )
        x = x.reshape(batch, 1, n_leaves * n_leaves).repeat(1, self.num_classes, 1)
        x = x.permute(0, 2, 1)  # (b, c, l, l)
        skip = x
        x = self.fc1_out(x)
        x = self.fc2_out(x)
        x = t.cat((x, skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)
        x = self.fc3_out(x)
        x = x.reshape(batch, n_leaves, n_leaves, self.num_classes)
        # x = x - b/self.num_classes/n_leaves
        # x = t.max(x, t.ones(x.shape)*(-1))

        # x = x*ed
        # x = x-ea

        x = x.permute(0, 3, 1, 2)  # (b, c, l, l)
        if self.symmetrize:
            x = t.div(x + t.transpose(x, 2, 3), 2)  # (b, c, l, l)

        return x
