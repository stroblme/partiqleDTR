import numpy as np
import time

import torch as t
from torch import nn
import torch.nn.functional as F

import mlflow

from .utils import *

import qiskit as q
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN

import logging

qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
from qiskit_machine_learning.connectors import TorchConnector

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob, batchnorm=True):
        super(MLP, self).__init__()

        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=True)
            # self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=False)  # Use this to overfit
        self.dropout_prob = do_prob

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
        x = F.elu(self.fc1(inputs))  # (b, l, d)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # (b, l, d)
        x = F.elu(self.fc2(x))  # (b, l, d)
        return self.batch_norm_layer(x) if self.batchnorm else x  # (b, l, d)


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
        n_blocks=3,
        dim_feedforward=128,  # ff
        n_layers_mlp=2,
        n_layers_vqc=3,
        n_additional_mlp_layers=2,
        n_final_mlp_layers=2,
        skip_block=True,
        skip_global=True,
        dropout_rate=0.3,
        factor=True,
        tokenize=-1,
        embedding_dims=-1,
        batchnorm=True,
        symmetrize=True,
        n_fsps: int = -1,
        device: str = "cpu",
        data_reupload=True,
        add_rot_gates=True,
        padding_dropout=True,
        mutually_exclusive_meas=True,
        **kwargs,
    ):
        super(qgnn, self).__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"

        self.num_classes = n_classes
        self.layers = n_layers_vqc  # dim_feedforward//8
        self.total_n_fsps = n_fsps  # used for padding in forward path
        self.factor = factor
        self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = n_additional_mlp_layers
        self.skip_block = skip_block
        self.skip_global = skip_global
        # self.max_leaves = max_leaves
        self.padding_dropout = padding_dropout
        self.mutually_exclusive_meas = mutually_exclusive_meas

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
                energy = q_params[i][3]

                px = (
                    q_params[i][0] * energy * t.pi,
                    i,
                    f"{identifier[:-3]}_rx_{i}",
                )
                qc.rx(*px)
                py = (
                    q_params[i][1] * energy * t.pi,
                    i,
                    f"{identifier[:-3]}_ry_{i}",
                )
                qc.ry(*py)
                pz = (
                    q_params[i][2] * energy * t.pi,
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






        # Set up embedding for tokens and adjust input dims
        if self.tokenize != -1:
            assert (embedding_dims is not None) and isinstance(
                embedding_dims, int
            ), "embedding_dims must be set to an integer is tokenize is given"

            # Initialise the embedding layers, ignoring pad values
            self.embed = nn.ModuleDict({})
            for idx, n_tokens in self.tokenize.items():
                # NOTE: This assumes a pad value of 0 for the input array x
                self.embed[str(idx)] = nn.Embedding(
                    n_tokens, embedding_dims, padding_idx=0
                )

            # And update the infeatures to include the embedded feature dims and delete the original, now tokenized feats
            n_momenta = n_momenta + (len(self.tokenize) * (embedding_dims - 1))
            print(f"Set up embedding for {len(self.tokenize)} inputs")

        # Create first half of inital NRI half-block to go from leaves to edges
        # initial_mlp = [
        #     MLP(n_momenta, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
        # ]
        initial_mlp = [
            MLP(1, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
        ]
        # Add any additional layers as per request
        initial_mlp.extend(
            [
                MLP(
                    dim_feedforward,
                    dim_feedforward,
                    dim_feedforward,
                    dropout_rate,
                    batchnorm,
                )
                for _ in range(n_layers_mlp - 1)
            ]
        )
        self.initial_mlp = nn.Sequential(*initial_mlp)

        # MLP to reduce feature dimensions from first Node2Edge before blocks begin
        self.pre_blocks_mlp = MLP(
            dim_feedforward * 2,
            dim_feedforward,
            dim_feedforward,
            dropout_rate,
            batchnorm,
        )

        block_dim = 3 * dim_feedforward if self.skip_block else 2 * dim_feedforward
        global_dim = 2 * dim_feedforward if self.skip_global else dim_feedforward

        if self.factor:
            # MLPs within NRI blocks
            # The blocks have minimum 1 MLP layer, and if specified they add more with a skip connection
            # List of blocks
            self.blocks = nn.ModuleList(
                [
                    # List of MLP sequences within each block
                    nn.ModuleList(
                        [
                            # MLP layers before Edge2Node (start of block)
                            nn.ModuleList(
                                [
                                    MLP(
                                        dim_feedforward,
                                        dim_feedforward,
                                        dim_feedforward,
                                        dropout_rate,
                                        batchnorm,
                                    ),
                                    nn.Sequential(
                                        *[
                                            MLP(
                                                dim_feedforward,
                                                dim_feedforward,
                                                dim_feedforward,
                                                dropout_rate,
                                                batchnorm,
                                            )
                                            for _ in range(n_additional_mlp_layers)
                                        ]
                                    ),
                                    # This is what would be needed for a concat instead of addition of the skip connection
                                    # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                                ]
                            ),
                            # MLP layers between Edge2Node and Node2Edge (middle of block)
                            nn.ModuleList(
                                [
                                    MLP(
                                        dim_feedforward,
                                        dim_feedforward,
                                        dim_feedforward,
                                        dropout_rate,
                                        batchnorm,
                                    ),
                                    nn.Sequential(
                                        *[
                                            MLP(
                                                dim_feedforward,
                                                dim_feedforward,
                                                dim_feedforward,
                                                dropout_rate,
                                                batchnorm,
                                            )
                                            for _ in range(n_additional_mlp_layers)
                                        ]
                                    ),
                                    # This is what would be needed for a concat instead of addition of the skip connection
                                    # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                                ]
                            ),
                            # MLP layer after Node2Edge (end of block)
                            # This is just to reduce feature dim after skip connection was concatenated
                            MLP(
                                block_dim,
                                dim_feedforward,
                                dim_feedforward,
                                dropout_rate,
                                batchnorm,
                            ),
                        ]
                    )
                    for _ in range(n_blocks)
                ]
            )
            print("Using factor graph MLP encoder.")
        else:
            self.mlp3 = MLP(
                dim_feedforward,
                dim_feedforward,
                dim_feedforward,
                dropout_rate,
                batchnorm,
            )
            self.mlp4 = MLP(
                dim_feedforward * 2,
                dim_feedforward,
                dim_feedforward,
                dropout_rate,
                batchnorm,
            )
            print("Using MLP encoder.")

        # Final linear layers as requested
        # self.final_mlp = nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers)])
        final_mlp = [
            MLP(global_dim, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
        ]
        # Add any additional layers as per request
        final_mlp.extend(
            [
                MLP(
                    dim_feedforward,
                    dim_feedforward,
                    dim_feedforward,
                    dropout_rate,
                    batchnorm,
                )
                for _ in range(n_final_mlp_layers - 1)
            ]
        )
        self.final_mlp = nn.Sequential(*final_mlp)

        self.fc_out = nn.Linear(dim_feedforward, self.num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

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
        # inputs=inputs.view(inputs.size(1), inputs.size(0), -1)

        if isinstance(inputs, (list, tuple)):
            inputs, rel_rec, rel_send = inputs
        else:
            rel_rec = None
            rel_send = None

        n_leaves, batch, feats = inputs.size()
        device = inputs.device

        # NOTE create rel matrices on the fly if not given as input
        if rel_rec is None:
            # rel_rec = t.eye(
            #     n_leaves,
            #     device=device
            # ).repeat_interleave(n_leaves-1, dim=1).T  # (l*(l-1), l)
            # rel_rec = rel_rec.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_rec = construct_rel_recvs([inputs.size(0)], device=device)

        if rel_send is None:
            # rel_send = t.eye(n_leaves, device=device).repeat(n_leaves, 1)
            # rel_send[t.arange(0, n_leaves*n_leaves, n_leaves + 1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*(l-1), l)
            # rel_send = rel_send.unsqueeze(0).expand(inputs.size(1), -1, -1)
            rel_send = construct_rel_sends([inputs.size(0)], device=device)

        # Input shape: [batch, num_atoms, num_timesteps, num_dims]
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # Need to match expected shape
        # TODO should batch_first be a dataset parameter?
        # (l, b, m) -> (b, l, m)
        x = inputs.permute(1, 0, 2)  # (b, l, m)

        # Create embeddings and merge back into x
        # TODO: Move mask creation to init, optimise this loop
        if self.tokenize != -1:
            emb_x = []
            # We'll use this to drop tokenized features from x
            mask = t.ones(feats, dtype=t.bool, device=device)
            for idx, emb in self.embed.items():
                # Note we need to convert tokens to type long here for embedding layer
                emb_x.append(emb(x[..., int(idx)].long()))  # List of (b, l, emb_dim)
                mask[int(idx)] = False

            # Now merge the embedding outputs with x (mask has deleted the old tokenized feats)
            x = t.cat([x[..., mask], *emb_x], dim=-1)  # (b, l, d + embeddings)
            del emb_x



        x = x.reshape(batch, n_leaves * feats)  # flatten the last two dims
        x = t.nn.functional.pad(
            x, (0, (self.total_n_fsps * feats) - (n_leaves * feats)), mode="constant", value=0
        )  # pad up to the largest lcag size. subtract the current size to prevent unnecessary padding.
        # note that x.size() is here n_leaves * feats

        # set the weights to zero which are either involved in a controlled operation or directly operating on qubits not relevant to the current graph (i.e. where the input was zero padded before)
        # this is supposed to ensure, that the actual measurement of the circuit is not impacted by any random weights contributing to meaningless 
        # print(self.quantum_layer._weights)
        if self.padding_dropout:
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
            permutations_indices = []
            for i in range(2**digits):
                if bin(i).count("1") == 1:
                    permutations_indices.append(i)
            assert len(permutations_indices) == digits
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
                lcag[i] = result[
                    i, permutations_indices
                ]
            # lcag[:, np.tril_indices_from(lcag[:], k=-1)] = result[:, permutations_indices]
            return lcag

        def get_all_shots(result, out_shape):
            lcag = t.zeros(out_shape)
            lcag = lcag.to(result.device)
            for i in range(out_shape[0]):
                lcag[i] = result[
                    i, out_shape[1]
                ]
            return lcag

        if self.mutually_exclusive_meas:
            x = get_binary_shots(
                x, build_binary_permutation_indices(n_leaves), (batch, n_leaves)
            )

            x = x.reshape(batch, n_leaves, 1)
        else:
            x = get_all_shots(
                x, (batch, 2**n_leaves-1)
            )

            x = x.reshape(batch, 1, 2**n_leaves-1).repeat(
                1, n_leaves, 1
            )

        # Initial set of linear layers
        # (b, l, 1) -> (b, l, d)
        x = self.initial_mlp(
            x
        )  # Series of 2-layer ELU net per node  (b, l, d) optionally includes embeddings

        # (b, l, d), (b, l*l, l), (b, l*l, l) -> (b, l, 2*d)
        x = self.node2edge(x, rel_rec, rel_send)  # (b, l*l, 2d)

        # All things related to NRI blocks are in here
        if self.factor:
            x = self.pre_blocks_mlp(x)  # (b, l*l, d)

            # Skip connection to jump over all NRI blocks
            x_global_skip = x

            for block in self.blocks:
                x_skip = x  # (b, l*l, d)

                # First MLP sequence
                x = block[0][0](x)  # (b, l*l, d)
                if self.block_additional_mlp_layers > 0:
                    x_first_skip = x  # (b, l*l, d)
                    x = block[0][1](x)  # (b, l*l, d)
                    x = x + x_first_skip  # (b, l*l, d)
                    del x_first_skip

                # Create nodes from edges
                x = self.edge2node(x, rel_rec)  # (b, l, d)

                # Second MLP sequence
                x = block[1][0](x)  # (b, l, d)
                if self.block_additional_mlp_layers > 0:
                    x_second_skip = x  # (b, l*l, d)
                    x = block[1][1](x)  # (b, l*l, d)
                    x = x + x_second_skip  # (b, l*l, d)
                    del x_second_skip

                # Create edges from nodes
                x = self.node2edge(x, rel_rec, rel_send)  # (b, l*l, 2d)

                if self.skip_block:
                    # Final MLP in block to reduce dimensions again
                    x = t.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*l, 3d)
                x = block[2](x)  # (b, l*l, d)
                del x_skip

            if self.skip_global:
                # Global skip connection
                x = t.cat(
                    (x, x_global_skip), dim=2
                )  # Skip connection  # (b, l*(l-1), 2d)

            # Cleanup
            del rel_rec, rel_send

        # else:
        #     x = self.mlp3(x)  # (b, l*(l-1), d)
        #     x = t.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)
        #     x = self.mlp4(x)  # (b, l*(l-1), d)

        # Final set of linear layers
        x = self.final_mlp(x)  # Series of 2-layer ELU net per node (b, l, d)

        # Output what will be used for LCA
        x = self.fc_out(x)  # (b, l*l, c)
        out = x.reshape(batch, n_leaves, n_leaves, self.num_classes)

        # Change to LCA shape
        # x is currently the flattened rows of the predicted LCA but without the diagonal
        # We need to create an empty LCA then populate the off-diagonals with their corresponding values
        # out = t.zeros((batch, n_leaves, n_leaves, self.num_classes), device=device)  # (b, l, l, c)
        # ind_upper = t.triu_indices(n_leaves, n_leaves, offset=1)
        # ind_lower = t.tril_indices(n_leaves, n_leaves, offset=-1)

        # Assign the values to their corresponding position in the LCA
        # The right side is just some quick mafs to get the correct edge predictions from the flattened x array
        # out[:, ind_upper[0], ind_upper[1], :] = x[:, (ind_upper[0] * (n_leaves - 1)) + ind_upper[1] - 1, :]
        # out[:, ind_lower[0], ind_lower[1], :] = x[:, (ind_lower[0] * (n_leaves - 1)) + ind_lower[1], :]

        # Need in the order for cross entropy loss
        out = out.permute(0, 3, 1, 2)  # (b, c, l, l)

        # Symmetrize
        if self.symmetrize:
            out = t.div(out + t.transpose(out, 2, 3), 2)  # (b, c, l, l)

        return out
