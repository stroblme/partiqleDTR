import numpy as np
import time

import torch as t
from torch import nn
import torch.nn.functional as F

import mlflow

from .utils import *
from .circuits import pqc_circuits, iec_circuits, circuit_builder
from .nri_blocks import MLP, generate_nri_blocks

import qiskit as q
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN, SamplerQNN

from qiskit.primitives import BackendSampler
from qiskit.providers.aer import QasmSimulator

import logging

qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
log = logging.getLogger(__name__)
from qiskit_machine_learning.connectors import TorchConnector

os.environ["CUDA_VISIBLE_DEVICES"] = ""




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
        padding_dropout=True,
        predefined_iec="",
        predefined_vqc="",
        measurement="entangled",
        backend="aer_simulator_statevector",
        **kwargs,
    ):
        super(qgnn, self).__init__()

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"

        self.start=None
        self.num_classes = n_classes
        self.layers = n_layers_vqc  # dim_feedforward//8
        self.total_n_fsps = n_fsps  # used for padding in forward path
        # self.factor = factor
        # self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = n_additional_mlp_layers
        self.skip_block = skip_block
        self.skip_global = skip_global
        # self.max_leaves = max_leaves
        self.padding_dropout = padding_dropout
        self.measurement = measurement
        self.predefined_vqc = predefined_vqc
        self.predefined_iec = predefined_iec
        self.data_reupload = data_reupload

        if "simulator" not in backend:
            from .ibmq_access import token, hub, group, project
            log.info(f"Searching for backend {backend} on IBMQ using token {token[:10]}****, hub {hub}, group {group} and project {project}")
            self.provider = q.IBMQ.enable_account(
                token=token,
                hub=hub,
                group=group,
                project=project,
            )
            self.backend = self.provider.get_backend(backend)
        else:
            log.info(f"Using simulator backend {backend}")
            self.backend = q.Aer.get_backend(backend)
            # # self.backend = QasmSimulator()
            # self.bs = BackendSampler(
            #     self.backend,
            #     # options={
            #     #     "shots": 2048,
            #     # },
            #     skip_transpilation=False,
            # )

        # self.qi = q.utils.QuantumInstance(
        #     self.backend
        # )

        self.enc_params = []
        self.var_params = []

        self.device = t.device(
            "cuda" if t.cuda.is_available() and device != "cpu" else "cpu"
        )

        log.info(
            f"Building Quantum Circuit with {self.layers} layers and {n_fsps} qubits"
        )

        self.qc = q.QuantumCircuit(n_fsps)
        circuit_builder(self.qc, self.predefined_iec, self.predefined_vqc, n_fsps, self.layers, data_reupload=self.data_reupload)

        mlflow.log_figure(self.qc.draw(output="mpl"), f"circuit.png")

        for param in self.qc.parameters:
            if "enc" in param.name:
                self.enc_params.append(param)
            else:
                self.var_params.append(param)

        log.info(
            f"Encoding Parameters: {len(self.enc_params)}, Variational Parameters: {len(self.var_params)}"
        )

        start = time.time()

        self.qnn = SamplerQNN(
            circuit=self.qc,
            # sampler=self.bs,
            input_params=self.enc_params,
            weight_params=self.var_params,
            # quantum_instance=self.qi,
            # interpret=interpreter,
            # output_shape=(n_fsps**2, n_classes),
            # sampling=True,
            input_gradients=True, #set to true as we are using torch connector
        )
        
        self.initial_weights = 2 * np.pi * q.utils.algorithm_globals.random.random(self.qnn.num_weights) - np.pi
        
        self.quantum_layer = TorchConnector(
            self.qnn, initial_weights=self.initial_weights
        )


        log.info(f"Transpilation took {time.time() - start}")
        log.info(f"Initialization done")

        # Create first half of inital NRI half-block to go from leaves to edges
        # initial_mlp = [
        #     MLP(n_momenta, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
        # ]
        if self.measurement == "mutually_exclusive":
            initial_mlp = [
                MLP(1, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
            ]
        elif self.measurement == "all":
            initial_mlp = [
                MLP(2**self.total_n_fsps, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
            ]
        elif self.measurement == "entangled":
            initial_mlp = [
                MLP(2**(self.total_n_fsps-1)+1, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
            ]
        else:
            raise ValueError("Invalid measurement specified")
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

        # calculate the dimensionality after each block and after all blocks
        # dimensionality increases if we introduce skip connections as one additional 
        # input is involved then
        block_dim = 3 * dim_feedforward if self.skip_block else 2 * dim_feedforward
        global_dim = 2 * dim_feedforward if self.skip_global else dim_feedforward

        
        self.blocks = generate_nri_blocks(dim_feedforward, batchnorm, dropout_rate, n_additional_mlp_layers, block_dim, n_blocks)

        print("Using factor graph MLP encoder.")

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
            rel_rec = construct_rel_recvs([inputs.size(0)], device=device)

        if rel_send is None:
            rel_send = construct_rel_sends([inputs.size(0)], device=device)

        # (l, b, m) -> (b, l, m)
        x = inputs.permute(1, 0, 2)  # (b, l, m)

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

        if self.start is not None:
            print(f"Duration: {time.time()-self.start}")
        self.start = time.time()
        


        if self.measurement == "mutually_exclusive":
            x = get_binary_shots(
                x, build_binary_permutation_indices(n_leaves), (batch, n_leaves)
            )

            x = x.reshape(batch, n_leaves, 1)
        elif self.measurement == "all":
            x = get_all_shots(
                x, (batch, 2**self.total_n_fsps)
            )

            x = x.reshape(batch, 1, 2**self.total_n_fsps).repeat(
                1, n_leaves, 1
            )
            # x = x.permute(0, 2, 1)  # (b, c, l, l) -> split the leaves
        elif self.measurement == "entangled":
            x = get_related_shots(
                x, build_related_permutation_indices(self.total_n_fsps), (batch, n_leaves, 2**(self.total_n_fsps-1)+1)
            )
        else:
            raise ValueError("Invalid measurement specified")


        x = self.forward_nri(x, rel_rec, rel_send)

        # Output what will be used for LCA
        x = self.fc_out(x)  # (b, l*l, c)
        x = x.reshape(batch, n_leaves, n_leaves, self.num_classes)

        # Need in the order for cross entropy loss
        x = x.permute(0, 3, 1, 2)  # (b, c, l, l)

        # Symmetrize
        if self.symmetrize:
            x = t.div(x + t.transpose(x, 2, 3), 2)  # (b, c, l, l)

        return x

    def forward_nri(self, x, rel_rec, rel_send):
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

        # Final set of linear layers
        x = self.final_mlp(x)  # Series of 2-layer ELU net per node (b, l, d)

        return x