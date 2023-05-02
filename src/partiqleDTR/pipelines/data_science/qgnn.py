import numpy as np
import time

import torch as t
from torch import nn

import mlflow

from .utils import *
from .circuits import circuit_builder, QuantumCircuit

from .gnn import gnn

from .custom_sampler_qnn import CustomSamplerQNN

import qiskit as q
from qiskit_aer import AerSimulator
from qiskit.visualization import *
from qiskit.primitives import BackendSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_ibm_provider import IBMProvider

from dask.distributed import LocalCluster, Client
from concurrent.futures import ThreadPoolExecutor

import logging

qiskit_logger = logging.getLogger("qiskit")
qiskit_logger.setLevel(logging.WARNING)
log = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class qgnn(nn.Module):
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
        n_shots=2048,
        initialization_constant=1,
        initialization_offset=0,
        parameter_seed=1111,
        **kwargs,
    ):
        nn.Module.__init__(self)

        assert dim_feedforward % 2 == 0, "dim_feedforward must be an even number"
        assert n_momenta == 4, "only supporting 4 momenta"

        self.evaluation_timestamp = None
        self.num_classes = n_classes
        self.layers = n_layers_vqc  # dim_feedforward//8
        self.total_n_fsps = n_fsps  # used for padding in forward path
        self.n_shots = n_shots
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = n_additional_mlp_layers
        self.skip_block = skip_block
        self.skip_global = skip_global
        self.padding_dropout = padding_dropout
        self.measurement = measurement
        self.predefined_vqc = predefined_vqc
        self.predefined_iec = predefined_iec
        self.data_reupload = data_reupload
        self.initialization_constant = initialization_constant
        self.param_rng = np.random.default_rng(seed=parameter_seed)

        self.device = t.device(
            "cuda" if t.cuda.is_available() and device != "cpu" else "cpu"
        )

        log.info(
            f"Building Quantum Circuit with {self.layers} layers and {n_fsps} qubits"
        )

        self.qc = QuantumCircuit(n_fsps)
        self.qc = circuit_builder(
            self.qc,
            self.predefined_iec,
            self.predefined_vqc,
            n_fsps,
            self.layers,
            data_reupload=self.data_reupload,
        )

        mlflow.log_figure(self.qc.draw(output="mpl"), f"circuit.png")

        self.enc_params = []
        self.var_params = []
        for param in self.qc.parameters:
            if "enc" in param.name:
                self.enc_params.append(param)
            else:
                self.var_params.append(param)

        log.info(
            f"Encoding Parameters: {len(self.enc_params)}, Variational Parameters: {len(self.var_params)}"
        )

        start = time.time()

        if "simulator" in backend:
            log.info(f"Using simulator backend {backend}")
            self.backend = q.Aer.get_backend(backend)
            # # self.backend = QasmSimulator()

            # n_workers = 5
            # exc = Client(address=LocalCluster(n_workers=n_workers, processes=True))
            # # exc = ThreadPoolExecutor(max_workers=n_workers)
            # # Set executor and max_job_size
            # self.backend.set_options(executor=exc)
            # self.backend.set_options(max_job_size=1) # see doc: https://qiskit.org/documentation/apidoc/parallel.html#usage-of-executor

            bs = BackendSampler(
                self.backend,
                options={
                    "shots": self.n_shots,
                },
            )
        elif "fake" in backend:
            backend = backend.replace("fake_", "")
            from .ibmq_access import token, hub, group, project

            log.info(
                f"Searching for backend {backend} on IBMQ using token {token[:10]}****, hub {hub}, group {group} and project {project}"
            )
            try:
                self.provider = IBMProvider(token=token, instance=f"{hub}/{group}/{project}")
            except:
                log.error("Failed to load accounts")
                raise RuntimeError

            device_backend = self.provider.get_backend(backend)
            self.backend = AerSimulator.from_backend(device_backend)

            bs = BackendSampler(
                self.backend,
                options={
                    "shots": self.n_shots,
                },
            )
        else:
            from .ibmq_access import token, hub, group, project

            log.info(
                f"Searching for backend {backend} on IBMQ using token {token[:10]}****, hub {hub}, group {group} and project {project}"
            )
            try:
                self.provider = IBMProvider(token=token, instance=f"{hub}/{group}/{project}")
            except:
                log.error("Failed to load accounts")
                raise RuntimeError
                
            self.backend = self.provider.get_backend(backend)

        qnn = CustomSamplerQNN(
            circuit=self.qc,
            sampler=bs,
            input_params=self.enc_params,
            weight_params=self.var_params,
            # quantum_instance=self.qi,
            # interpret=interpreter,
            # output_shape=(n_fsps**2, n_classes),
            input_gradients=True,  # set to true as we are using torch connector
        )

        if type(self.initialization_constant) == str:
            if self.initialization_constant == "strategy_a":
                # https://arxiv.org/abs/2302.06858
                L = self.qc.depth()
                raise NotImplementedError()

        self.initial_weights = (
            self.initialization_constant
            * 2 * np.pi #times 2 so that we don't end up with [-a/2 *pi..a/2 *pi]
            * self.param_rng.random(qnn.num_weights)
            - (self.initialization_constant) * np.pi
            + initialization_offset * np.pi
        ) # [-(a*pi + b) .. (a*pi + b)]

        self.quantum_layer = TorchConnector(qnn, initial_weights=self.initial_weights)

        log.info(f"Transpilation took {time.time() - start}")
        log.info(f"Initialization done")

        # Create first half of inital NRI half-block to go from leaves to edges
        # initial_mlp = [
        #     MLP(n_momenta, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
        # ]
        if self.measurement == "mutually_exclusive":
            n_input = 1
        elif self.measurement == "all":
            n_input = 2**self.total_n_fsps
        elif self.measurement == "entangled":
            n_input = 2 ** (self.total_n_fsps - 1) + 1
        else:
            raise ValueError("Invalid measurement specified")


        self.gnn = gnn(
                        n_input,  # d
                        n_classes,  # l
                        n_blocks,
                        dim_feedforward,  # ff
                        n_layers_mlp,
                        n_additional_mlp_layers,
                        n_final_mlp_layers,
                        skip_block,
                        skip_global,
                        dropout_rate,
                        batchnorm,
                        symmetrize,
        )


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

        x = x.reshape(batch, n_leaves * feats)  # flatten the last two dims
        x = t.nn.functional.pad(
            x,
            (0, (self.total_n_fsps * feats) - (n_leaves * feats)),
            mode="constant",
            value=0,
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

        # if self.evaluation_timestamp is not None:
        #     print(f"Duration: {time.time()-self.evaluation_timestamp}")
        # self.evaluation_timestamp = time.time()

        if self.measurement == "mutually_exclusive":
            x = get_binary_shots(
                x, build_binary_permutation_indices(n_leaves), (batch, n_leaves)
            )

            x = x.reshape(batch, n_leaves, 1)
        elif self.measurement == "all":
            x = get_all_shots(x, (batch, 2**self.total_n_fsps))

            x = x.reshape(batch, 1, 2**self.total_n_fsps).repeat(1, n_leaves, 1)
            # x = x.permute(0, 2, 1)  # (b, c, l, l) -> split the leaves
        elif self.measurement == "entangled":
            x = get_related_shots(
                x,
                build_related_permutation_indices(self.total_n_fsps),
                (batch, n_leaves, 2 ** (self.total_n_fsps - 1) + 1),
            )
        else:
            raise ValueError("Invalid measurement specified")

        # The whole rest of the architecture
        x = self.gnn.forward_nri(x, rel_rec, rel_send)

        # Output what will be used for LCA
        x = self.gnn.fc_out(x)  # (b, l*l, c)
        x = x.reshape(batch, n_leaves, n_leaves, self.num_classes)

        # Required for cross entropy loss
        x = x.permute(0, 3, 1, 2)  # (b, c, l, l)

        # Symmetrize
        if self.symmetrize:
            x = t.div(x + t.transpose(x, 2, 3), 2)  # (b, c, l, l)

        return x


    def print_layer(x):
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots

        rows = x.shape[0]
        fig = make_subplots(rows=8, cols=1)

        for i in range(rows):
            fig.add_trace(go.Heatmap(
                    z=x[i],
                    type="heatmap",
                ),
                col=1, row=i+1)

        fig.show()