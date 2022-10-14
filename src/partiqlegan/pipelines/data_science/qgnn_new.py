import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
import time

from .graph_visualization import hierarchy_pos

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

import networkx as nx
import torch_geometric as tg
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import add_self_loops, degree
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


class qMessagePassing(tg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow='target_to_source')  # "Add" aggregation (Step 5).
        self.lin = t.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = t.nn.Parameter(t.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm) # internally calls message, aggregate, update

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class qgnn(nn.Module):
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
        dim_feedforward=6,  # ff
        n_layers_mlp=2,
        n_layers_vqc=3,
        skip_block=False,
        dropout_rate=0.3,
        batchnorm=True,
        symmetrize=True,
        n_fsps: int = -1,
        device: str = "cpu",
        data_reupload=True,
        add_rot_gates=True,
        **kwargs,
    ):
        super(qgnn, self).__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"
        # n_fsps = 4
        self.layers = n_layers_vqc  # dim_feedforward//8
        self.total_n_fsps = n_fsps  # used for padding in forward path
        self.num_classes = n_classes
        self.symmetrize = symmetrize
        self.skip_block = skip_block
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
                if add_rot_gates:
                    qc.rx(
                        q.circuit.Parameter(f"{identifier}_rx_0_{i}"),
                        i,
                        f"{identifier}_rx_{i}",
                    )
                    qc.ry(
                        q.circuit.Parameter(f"{identifier}_ry_0_{i}"),
                        i,
                        f"{identifier}_ry_{i}",
                    )
                    qc.rz(
                        q.circuit.Parameter(f"{identifier}_rz_0_{i}"),
                        i,
                    )
                if i == 0:
                    qc.crx(
                        q.circuit.Parameter(f"{identifier}_crx_{n_qubits - 1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_crx_{n_qubits - 1}_{i}",
                    )
                    qc.cry(
                        q.circuit.Parameter(f"{identifier}_cry_{n_qubits - 1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_cry_{n_qubits - 1}_{i}",
                    )
                    qc.crz(
                        q.circuit.Parameter(f"{identifier}_crz_{n_qubits - 1}_{i}"),
                        i,
                        n_qubits - 1,
                        f"{identifier}_crz_{n_qubits - 1}_{i}",
                    )
                else:
                    qc.crx(
                        q.circuit.Parameter(f"{identifier}_crx_{n_qubits - i - 1}_{n_qubits - i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_crx_{n_qubits - i - 1}_{n_qubits - i}",
                    )
                    qc.cry(
                        q.circuit.Parameter(f"{identifier}_cry_{n_qubits - i - 1}_{n_qubits - i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_cry_{n_qubits - i - 1}_{n_qubits - i}",
                    )
                    qc.crz(
                        q.circuit.Parameter(f"{identifier}_crz_{n_qubits - i - 1}_{n_qubits - i}"),
                        n_qubits - i,
                        n_qubits - i - 1,
                        f"{identifier}_crz_{n_qubits - i - 1}_{n_qubits - i}",
                    )

        def circuit_builder(qc, n_qubits, n_hidden):
            enc_params = gen_encoding_params(n_qubits, f"enc")
            for i in range(n_hidden):
                if data_reupload or i == 0:
                    encoding(qc, n_qubits, enc_params, f"enc_{i}")
                qc.barrier()
                variational(qc, n_qubits, f"var_{i}")
                qc.barrier()

        log.info(
            f"Building Quantum Circuit with {self.layers} layers and {n_fsps} qubits"
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

        # start = time.time()
        # self.qnn = CircuitQNN(
        #     self.qc,
        #     self.enc_params,
        #     self.var_params,
        #     quantum_instance=self.qi,
        #     # interpret=interpreter,
        #     # output_shape=(n_fsps**2, n_classes),
        #     # sampling=True,
        #     input_gradients=True,
        # )
        # self.initial_weights = 0.1 * (
        #     2 * q.utils.algorithm_globals.random.random(self.qnn.num_weights) - 1
        # )
        # log.info(f"Transpilation took {time.time() - start}")
        # self.quantum_layer = TorchConnector(
        #     self.qnn, initial_weights=self.initial_weights
        # )
        # log.info(f"Initialization done")

        # last layer input size depends if we do a concat before (use skip cons)
        final_input_dim = 2 * dim_feedforward if self.skip_block else dim_feedforward

        self.block = nn.ModuleList(
            [
                MLP(
                    self.num_classes,
                    dim_feedforward,
                    dim_feedforward,
                    dropout_rate,
                    batchnorm,
                    activation=F.elu,
                    device=self.device,
                ),
                nn.Sequential(
                    *[
                        MLP(
                            dim_feedforward,
                            dim_feedforward,
                            dim_feedforward,
                            dropout_rate,
                            batchnorm,
                            activation=F.elu,
                            device=self.device,
                        )
                        for _ in range(n_layers_mlp)
                    ]
                ),
                MLP(
                    final_input_dim,
                    final_input_dim // 2,
                    self.num_classes,
                    dropout_rate,
                    batchnorm,
                    activation=F.elu,
                    device=self.device,
                )
                # This is what would be needed for a concat instead of addition of the skip connection
                # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
            ]
        )

        self.qgat = qMessagePassing(4, 4)

    def forward(self, inputs):
        """
        Input: (l, b, d)
        Output: (b, c, l, l)
        """
        if isinstance(inputs, (list, tuple)):
            (
                x,
                rel_rec,
                rel_send,
            ) = inputs  # get the actual state from the list of states, rel_rec and rel_send
        else:  # in case we are running torchinfo, rel_rec and rel_send are not provided
            rel_rec = None
            rel_send = None
            x = inputs

        n_leaves, batch, feats = x.size()  # get the representative sizes
        assert x.device == self.device  # check if we are running on the same device

        x = x.permute(1, 0, 2)  # (b, l, f)

        def build_fully_connected(input):
            batch, n_leaves, n_feats = input.size()

            max_nodes = np.sum(range(n_leaves+1))
            graph_list = []
            for data_batch in input:

                edges = []
                for level in range(n_leaves-1):
                    for n_level in range(level+1):
                        level_node_idx = n_level + (2**level-1)
                        max_level_node_idx = level + (2**level-1)
                        for n_other in range(max_level_node_idx+1, max_nodes):
                            edges.append([level_node_idx, n_other])

                edges = t.Tensor(edges).long().permute(1,0)
                embeddings = t.zeros(max_nodes, n_feats)
                embeddings[-n_leaves:] = data_batch
                # edges = t.combinations(t.Tensor(range(n_leaves)).long()).permute(1,0)
                graph_batch = tg.data.Data(x=embeddings, edge_index=edges)

                graph_list.append(graph_batch)

            return tg.data.Batch().from_data_list(graph_list)

        x = build_fully_connected(x)

        vis = to_networkx(x)
        plt.close()
        plt.figure(1,figsize=(8,8)) 
        pos=nx.circular_layout(vis)
        # pos = hierarchy_pos(vis, max(max(x.edge_index)))
        nx.draw(vis, pos, cmap=plt.get_cmap('Set3'),node_size=70,linewidths=6)
        plt.savefig("graph.png")


        x.x = self.qgat(x.x, x.edge_index) # messages flow to uninitialized nodes in graph

        x = x.reshape(batch, n_leaves * feats)  # flatten the last two dims
        x = t.nn.functional.pad(
            x, (0, (self.total_n_fsps * feats) - (n_leaves * feats)), mode="constant", value=0
        )  # pad up to the largest lcag size. subtract the current size to prevent unnecessary padding.
        # note that x.size() is here n_leaves * feats

        # set the weights to zero which are either involved in a controlled operation or directly operating on qubits not relevant to the current graph (i.e. where the input was zero padded before)
        # this is supposed to ensure, that the actual measurement of the circuit is not impacted by any random weights contributing to meaningless 
        # print(self.quantum_layer._weights)
        with t.no_grad():
            for i, p in enumerate(self.var_params):
                if int(p._name[-1]) > n_leaves or int(p._name[-3]) > n_leaves:
                    self.quantum_layer._weights[i] = 0.0
        # print(self.quantum_layer._weights)

        x = self.quantum_layer(x)

        

        def build_binary_permutation_indices(digits):
            """
            Generate the binary permutation indices.
            :param digits:
            :return:
            """
            n_permutations = (digits**2 - digits) // 2
            permutations_indices = []
            for i in range(2**digits):
                if bin(i).count("1") == 2:
                    permutations_indices.append(i)
            assert len(permutations_indices) == n_permutations
            return permutations_indices

        def get_binary_shots(result, permutations_indices, out_shape):
            """
            Generate the binary shots.
            :param result:
            :param permutations_indices:
            :param out_shape:
            :return:
            """
            lcag = t.zeros(out_shape)
            lcag = lcag.to(result.device)
            for i in range(out_shape[0]):
                lcag[i][np.tril_indices_from(lcag[0], k=-1)] = result[
                    i, permutations_indices
                ]
            # lcag[:, np.tril_indices_from(lcag[:], k=-1)] = result[:, permutations_indices]
            lcag = lcag + t.transpose(lcag, 1, 2)
            return lcag

        x = get_binary_shots(
            x, build_binary_permutation_indices(n_leaves), (batch, n_leaves, n_leaves)
        )

        x = x.reshape(batch, 1, n_leaves * n_leaves).repeat(
            1, self.num_classes, 1
        )  # copy the vqc output across n_classes dimensions and let the nn handle the decoding
        x = x.permute(0, 2, 1)  # (b, c, l, l) -> split the leaves

        skip = x if self.skip_block else None
        x = self.block[0](x)  # initial mlp
        for seq_mlp in self.block[1]:
            x = seq_mlp(x)

        x = t.cat((x, skip), dim=2) if self.skip_block else x  # Skip connection
        x = self.block[2](x)  # (b, c, l, l) -> final mlp

        x = x.reshape(batch, n_leaves, n_leaves, self.num_classes)
        x = x.permute(0, 3, 1, 2)  # (b, c, l, l)

        x = t.div(x + t.transpose(x, 2, 3), 2) if self.symmetrize else x  # (b, c, l, l)

        return x
