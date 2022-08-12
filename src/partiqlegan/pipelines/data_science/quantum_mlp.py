import time

import torch.nn.functional as F

import mlflow

from .utils import *

import qiskit as q
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector



class quantum_mlp():

    def __init__(self, n_in, n_hid, n_out, do_prob, batchnorm=True, activation=F.elu):

        self.qi = q.utils.QuantumInstance(q.Aer.get_backend('aer_simulator_statevector'))

        self.enc_params = []
        self.var_params = []
        def dataReupload(qc, n_qubits, identifier):
            for i in range(n_qubits**2):
                param = (q.circuit.Parameter(f"{identifier}_rxy_{i}"), i//n_qubits, f"{identifier}_rxy_{i}")
                qc.rx(*param) 
                # qc.ry(*param) 
                

        def mps(qc, n_qubits, identifier):
            for i in range(n_qubits):
                qc.ry(q.circuit.Parameter(f"{identifier}_ry_0_{i}"), i, f"{identifier}_ry_0_{i}") 
            
            for i in range(n_qubits-1):
                qc.swap(i, i+1)
                qc.ry(q.circuit.Parameter(f"{identifier}_ry_{i+1}_{i+1}"), i+1, f"{identifier}_ry_{i+1}_{i+1}")

        def build_circuit_19(qc, n_qubits, identifier):
            for i in range(n_qubits):
                qc.rx(q.circuit.Parameter(f"{identifier}_rx_0_{i}"), i, f"{identifier}_rx_0_{i}") 
                qc.rz(q.circuit.Parameter(f"{identifier}_rz_1_{i}"), i) 

            for i in range(n_qubits-1):
                if i == 0:
                    qc.crx(q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"), i, n_qubits-1, f"{identifier}_crx_{i+1}_{i}")
                else:
                    qc.crx(q.circuit.Parameter(f"{identifier}_crx_{i+1}_{i}"), n_qubits-i, n_qubits-i-1, f"{identifier}_crx_{i+1}_{i}")



        def circuit_builder(qc, n_qubits, n_hidden):
            for n_hid in range(n_hidden):
                dataReupload(qc, n_qubits, f"dru_{n_hid}")
                qc.barrier()
                if n_hid % 16 == 0:
                    build_circuit_19(qc, n_qubits, f"mps_{n_hid}")


        log.info(f"Building Quantum Circuit with {self.layers} layers and {n_classes} qubits")
        self.qc = q.QuantumCircuit(n_fsps)
        circuit_builder(self.qc, n_fsps, self.layers)

        mlflow.log_figure(self.qc.draw(output="mpl"), f"circuit.png")

        for param in self.qc.parameters:
            if "dru" in param.name:
                self.enc_params.append(param)
            else:
                self.var_params.append(param)
        log.info(f"Encoding Parameters: {len(self.enc_params)}, Variational Parameters: {len(self.var_params)}")

        def interpreter(x):
            print(f"Interpreter Input {x}")
            return x

        start = time.time()
        self.qnn = CircuitQNN(  self.qc,
                                self.enc_params, self.var_params,
                                quantum_instance=self.qi,
                                # interpret=interpreter,
                                # output_shape=(n_fsps**2, n_classes),
                                # sampling=True,
                                input_gradients=True
                            )
        self.initial_weights = 0.1 * (2 * q.utils.algorithm_globals.random.random(self.qnn.num_weights) - 1)
        log.info(f"Transpilation took {time.time() - start}")
        self.quantum_layer = TorchConnector(self.qnn, initial_weights=self.initial_weights)
        log.info(f"Initialization done")