import numpy as np
import matplotlib.pyplot as plt

from enum import Enum

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch as t
from torch import nn
import torch.nn.functional as F

from .utils import *

import qiskit as q
from qiskit import transpile, assemble
from qiskit.visualization import *

from multiprocessing import Pool

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


class CircuitType(Enum):
    EdgeNetwork = 1
    NodeNetwork = 2
class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """
    
    def __init__(self, n_qubits:int, circuit_type:CircuitType, backend, shots):
        # --- Circuit definition ---
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend
        self.circuit_type = circuit_type

        # parameters = [np.random.uniform(-np.pi, np.pi) for i in range(n_qubits*(n_layers[0]*len(rot_gates_enc)+n_layers[1]*len(rot_gates_pqc)+n_layers[1]*len(ent_gates_pqc)))]
        
        self.enc_qc = q.QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.enc_qc.ry(q.circuit.Parameter(f"enc_ry_0_{i}"), i, f"enc_ry_0_{i}")



        
        def build_mps_circuit(qc):
            for i in range(n_qubits):
                qc.ry(q.circuit.Parameter(f"var_ry_0_{i}"), i, f"var_ry_0_{i}") 
            
            for i in range(n_qubits-1):
                qc.swap(i, i+1)
                qc.ry(q.circuit.Parameter(f"var_ry_{i+1}_{i+1}"), i+1, f"var_ry_{i+1}_{i+1}")

        
        def build_circuit_19(qc:q.QuantumCircuit):
            for i in range(n_qubits):
                qc.rx(q.circuit.Parameter(f"var_rx_0_{i}"), i, f"var_rx_0_{i}") 
                qc.rz(q.circuit.Parameter(f"var_rz_1_{i}"), i) 
            
            for i in range(n_qubits-1):
                if i == 0:
                    qc.crx(q.circuit.Parameter(f"var_crx_{i+1}_{i}"), i, n_qubits-1, f"var_crx_{i+1}_{i}")
                else:
                    qc.crx(q.circuit.Parameter(f"var_crx_{i+1}_{i}"), n_qubits-i, n_qubits-i-1, f"var_crx_{i+1}_{i}")


        self.var_qc = q.QuantumCircuit(n_qubits) # no need to add classical since they are automaticallya added later
        if circuit_type==CircuitType.NodeNetwork:
            build_circuit_19(self.var_qc)
            self.qc = self.enc_qc.compose(self.var_qc)
        else:
            build_mps_circuit(self.var_qc)
            self.qc = self.enc_qc.compose(self.var_qc)
            # self.qc.add_bits(q.ClassicalRegister(1)) # add one classical bit here so we can do the measurement later
            # self.qc.measure(n_qubits-1, 0)
        self.qc.measure_all() # TODO: we need to measure all since somehow we get an error when evaluating a circuit where not *all* qubits are measured


        # self.qc.draw()
    
    def circuit_parameters(self, data, variational):
        parameters = {}
        for i, p in enumerate(self.enc_qc.parameters):
            parameters[p] = data[i]
        for i, p in enumerate(self.var_qc.parameters):
            parameters[p] = variational[i]
        return parameters

    def bitstring_decode(self, results):
        shots = sum(results.values())

        average = np.zeros(self.n_qubits) if self.circuit_type == CircuitType.NodeNetwork else np.zeros(1)
        for bitstring, counts in results.items():
            if self.circuit_type == CircuitType.NodeNetwork:
                for i, s in enumerate(bitstring):
                    average[i] += counts if s == "1" else 0
            else:
                average[0] += counts if bitstring[-1] == "1" else 0

        return average/(shots*len(average))

    def run(self, nd_data, variational:np.array):
        circuits = []
        ignore_after = -1
        for i, data in enumerate(nd_data):
            if max(data) == 0.0:
                ignore_after = i
                break # shortcut to prevent unnecessary circuit exec.
            else:
                scaled_data = np.interp(data, (data.min(), data.max()), (0, np.pi))
                circuits.append(self.qc.assign_parameters(self.circuit_parameters(scaled_data.tolist(), variational.tolist())))



        # backend = q.BasicAer.get_backend('qasm_simulator')
        jobs_result =  q.execute(circuits, self.backend, shots=self.shots).result()
        expectations = [self.bitstring_decode(jobs_result.get_counts(c)) for c in circuits]
        expectations = np.append(expectations, [np.zeros(4)]*(len(nd_data)-ignore_after), axis=0) if ignore_after >= 0 else expectations
        # counts = np.array(list(results.get_counts().values()))
        # states = np.array(list(results.get_counts().keys())).astype(float)
        
        return np.array(expectations)

# simulator = qiskit.Aer.get_backend('aer_simulator')

# circuit = QuantumCircuit(1, simulator, 100)
# print('Expected value for rotation pi {}'.format(circuit.run([np.pi])[0]))
# circuit._circuit.draw()
# def poolProcess(circuit_data_variational):
#     circuit, data, variational = circuit_data_variational
#     return circuit.run(data, variational)

class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, variational, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        ctx.variational = variational

        results = []
        # with Pool(len(input)) as p:
        #     results = p.map(poolProcess, list(zip([ctx.quantum_circuit]*len(input), input, [ctx.variational]*len(input))))
        # can't use pool processing here since qiskit itself has poolprocessing
        for batch in input:
            expectation_z = ctx.quantum_circuit.run(batch, ctx.variational)
            results.append(expectation_z)

        results = t.Tensor(results)

        ctx.save_for_backward(input, results)

        return results
        
    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, results = ctx.saved_tensors
        shift_right = ctx.variational + np.ones(ctx.variational.shape) * ctx.shift
        shift_left = ctx.variational - np.ones(ctx.variational.shape) * ctx.shift
        
        gradients = []
        for batch in input:

            expectation_right = ctx.quantum_circuit.run(batch, shift_right)
            expectation_left  = ctx.quantum_circuit.run(batch, shift_left)
            
            gradient = expectation_right - expectation_left # parmeter shift rule
            gradients.append(gradient)

        gradients = t.Tensor(gradients)

        return gradients.float() * grad_output.float()

class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, n_in, n_hid, n_out, circuit_type, backend=None, shots=100, shift=np.pi/2):
        super(Hybrid, self).__init__()
        if backend == None:
            backend = q.Aer.get_backend('aer_simulator')
        self.fc_in = nn.Linear(n_in, n_hid)
        if circuit_type == CircuitType.NodeNetwork:
            self.fc_out = nn.Linear(n_hid, n_out)
        else:
            self.fc_out = nn.Linear(1, n_out)

        self.quantum_circuit = QuantumCircuit(4, circuit_type, backend, shots)
        self.shift = shift

        self.variational = np.random.random(self.quantum_circuit.var_qc.num_parameters)
        
    def forward(self, input):
        x = self.fc_in(input)
        x = HybridFunction.apply(x, self.quantum_circuit, self.variational, self.shift)
        x = self.fc_out(x)
        return x

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
        '''
        Input: (b, l, c)
        Output: (b, l, d)
        '''
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))  # (b, l, d)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # (b, l, d)
        x = F.elu(self.fc2(x))  # (b, l, d)
        return self.batch_norm_layer(x) if self.batchnorm else x  # (b, l, d)


class qgnn(nn.Module):
    ''' NRI model built off the official implementation.

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
    '''
    def __init__(
        self,
        n_momenta, # d
        n_classes, # l
        n_blocks=3,
        dim_feedforward=128, # ff
        n_layers_mlp=2,
        n_additional_mlp_layers=2,
        n_final_mlp_layers=2,
        dropout_rate=0.3,
        factor=True,
        tokenize=-1,
        embedding_dims=-1,
        batchnorm=True,
        symmetrize=True,
        **kwargs,
    ):
        super(qgnn, self).__init__()

        assert dim_feedforward % 2 == 0, 'dim_feedforward must be an even number'

        self.num_classes = n_classes
        self.factor = factor
        self.tokenize = tokenize
        self.symmetrize = symmetrize
        self.block_additional_mlp_layers = n_additional_mlp_layers
        # self.max_leaves = max_leaves

        # Set up embedding for tokens and adjust input dims
        if self.tokenize != -1:
            assert (embedding_dims is not None) and isinstance(embedding_dims, int), 'embedding_dims must be set to an integer is tokenize is given'

            # Initialise the embedding layers, ignoring pad values
            self.embed = nn.ModuleDict({})
            for idx, n_tokens in self.tokenize.items():
                # NOTE: This assumes a pad value of 0 for the input array x
                self.embed[str(idx)] = nn.Embedding(n_tokens, embedding_dims, padding_idx=0)

            # And update the infeatures to include the embedded feature dims and delete the original, now tokenized feats
            n_momenta = n_momenta + (len(self.tokenize) * (embedding_dims - 1))
            print(f'Set up embedding for {len(self.tokenize)} inputs')

        # Create first half of inital NRI half-block to go from leaves to edges
        initial_mlp = [MLP(n_momenta, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)]
        # Add any additional layers as per request
        initial_mlp.extend([
            MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout_rate, batchnorm) for _ in range(n_layers_mlp - 1)
        ])
        self.initial_mlp = nn.Sequential(*initial_mlp)
        self.node_network = Hybrid(dim_feedforward, 4, dim_feedforward, CircuitType.NodeNetwork)
        self.edge_network = Hybrid(dim_feedforward, 4, 1, CircuitType.EdgeNetwork)

        # MLP to reduce feature dimensions from first Node2Edge before blocks begin
        self.pre_blocks_mlp = MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)

        if self.factor:
            # MLPs within NRI blocks
            # The blocks have minimum 1 MLP layer, and if specified they add more with a skip connection
            # List of blocks
            self.blocks = nn.ModuleList([
                # List of MLP sequences within each block
                nn.ModuleList([
                    # MLP layers before Edge2Node (start of block)
                    nn.ModuleList([
                        MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout_rate, batchnorm),
                        *[Hybrid(dim_feedforward, 4, 1, CircuitType.EdgeNetwork) for _ in range(self.num_classes)]
                        # This is what would be needed for a concat instead of addition of the skip connection
                        # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                    ]),
                    # MLP layers between Edge2Node and Node2Edge (middle of block)
                    nn.ModuleList([
                        MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout_rate, batchnorm),
                        Hybrid(dim_feedforward, 4, dim_feedforward, CircuitType.NodeNetwork)
                        # This is what would be needed for a concat instead of addition of the skip connection
                        # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                    ]),
                    # MLP layer after Node2Edge (end of block)
                    # This is just to reduce feature dim after skip connection was concatenated
                    MLP(dim_feedforward * 3, dim_feedforward, dim_feedforward, dropout_rate, batchnorm),
                ]) for _ in range(n_blocks)
            ])
            print("Using factor graph MLP encoder.")
        else:
            self.mlp3 = MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
            self.mlp4 = MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)
            print("Using MLP encoder.")

        # Final linear layers as requested
        # self.final_mlp = nn.Sequential(*[MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout, batchnorm) for _ in range(final_mlp_layers)])
        final_mlp = [MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout_rate, batchnorm)]
        # Add any additional layers as per request
        final_mlp.extend([
            MLP(dim_feedforward, dim_feedforward, dim_feedforward, dropout_rate, batchnorm) for _ in range(n_final_mlp_layers - 1)
        ])
        self.final_mlp = nn.Sequential(*final_mlp)

        self.fc_out = nn.Linear(dim_feedforward, self.num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec):
        '''
        Input: (b, l*l, d), (b, l*l, l)
        Output: (b, l, d)
        '''
        # TODO assumes that batched matrix product just works
        # TODO these do not have to be members
        incoming = t.matmul(rel_rec.permute(0, 2, 1), x)  # (b, l, d)
        denom = rel_rec.sum(1)[:, 1]
        return incoming / denom.reshape(-1, 1, 1)  # (b, l, d)
        # return incoming / incoming.size(1)  # (b, l, d)

    def node2edge(self, x, rel_rec, rel_send):
        '''
        Input: (b, l, d), (b, l*(l-1), l), (b, l*(l-1), l)
        Output: (b, l*l(l-1), 2d)
        '''
        # TODO assumes that batched matrix product just works
        receivers = t.matmul(rel_rec, x)  # (b, l*l, d)
        senders = t.matmul(rel_send, x)  # (b, l*l, d)
        edges = t.cat([senders, receivers], dim=2)  # (b, l*l, 2d)

        return edges

    def forward(self, inputs):
        '''
        Input: (l, b, d)
        Output: (b, c, l, l)
        '''
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

        # Initial set of linear layers
        # (b, l, m) -> (b, l, d)
        # x = self.hybrid(x)
        x = self.initial_mlp(x)  # Series of 2-layer ELU net per node  (b, l, d) optionally includes embeddings

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

                # Final MLP in block to reduce dimensions again
                x = t.cat((x, x_skip), dim=2)  # Skip connection  # (b, l*l, 3d)
                x = block[2](x)  # (b, l*l, d)
                del x_skip

            # Global skip connection
            x = t.cat((x, x_global_skip), dim=2)  # Skip connection  # (b, l*(l-1), 2d)

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
