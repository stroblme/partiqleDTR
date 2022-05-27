"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.7
"""

# import numpy as np
# from typing import Dict, Tuple, List
# from phasespace import GenParticle, nbody_decay
# from phasespace.fromdecay import GenMultiDecay
# import pandas as pd

# from qiskit import Aer, QuantumCircuit
# from qiskit.utils import QuantumInstance, algorithm_globals
# from qiskit.opflow import AerPauliExpectation
# from qiskit.circuit import Parameter
# from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
# from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
# from qiskit_machine_learning.connectors import TorchConnector

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable

# import torch.nn as nn
# import torch.nn.functional as F

import numpy as np
import cirq
import sympy
import json
import tensorflow as tf
import tensorflow_quantum as tfq

import qcircuits as qcircuits

class QCircuit:
    def __init__(self, circuits_metadata, IEC_id, PQC_id, MC_id, n_layers=1, input_size=4, p=None):
        self.n_layers = n_layers
        self.n_inputs = input_size

        self.IEC_id = IEC_id
        self.PQC_id = PQC_id
        self.MC_id  = MC_id
        self.p = p

        # read metadata
        # with open('qcircuits/circuits_metadata.json') as json_file:
        self.metadata = circuits_metadata

        self.n_qubits = self.get_n_qubits()
        self.n_params = self.get_n_params()
        self.n_measurements = self.get_measurements()

    def model_circuit(self):
        self.qubits  = cirq.GridQubit.rect(self.n_qubits, 1)
        self.circuit = cirq.Circuit()

        self.IEC()(self.circuit, self.qubits, n_qubits = self.n_qubits)
        self.PQC()(self.circuit, self.qubits, n_layers = self.n_layers, n_qubits = self.n_qubits)
        
        return self.circuit, self.qubits

    def pqc_circuit(self):
        self.qubits  = cirq.GridQubit.rect(self.n_qubits, 1)
        self.circuit = cirq.Circuit()

        self.PQC()(self.circuit, self.qubits, n_layers = self.n_layers, n_qubits = self.n_qubits)
        
        return self.circuit, self.qubits

    def IEC(self):
        '''information encoding circuit'''
        return getattr(qcircuits, self.metadata['qc_iec_dict'][self.IEC_id])
    def PQC(self):
        '''parametrized quantum circuit'''
        return getattr(qcircuits, self.metadata['qc_pqc_dict'][self.PQC_id])
    def measurement_operators(self):
        '''measurement block of the circuit'''
        
        self.qc_meas_dict = {
            'measure_all': qcircuits.measure_all,
            'measure_last': qcircuits.measure_last,
            #'none': qcircuits.state_vector,
            #'probs': qcircuits.probs,
            #'samples': qcircuits.sample,
            }
        return self.qc_meas_dict[self.MC_id](qubits=self.qubits, n_measurements=self.n_measurements)
        

    def get_n_params(self):
        if self.PQC_id in self.metadata['weight_shapes_dict'].keys():
            n_params  = self.metadata['weight_shapes_dict'][self.PQC_id]*self.n_layers
        elif self.PQC_id == '19':
            n_params =  3*self.n_qubits*self.n_layers
        elif self.PQC_id == '15':
            n_params =  2*self.n_qubits*self.n_layers
        elif self.PQC_id == '14':
            n_params = (3*self.n_qubits + self.n_qubits/np.gcd(self.n_qubits,3))*self.n_layers
        elif self.PQC_id == '10':
            n_params = self.n_qubits*(self.n_layers+1)
        elif self.PQC_id == '10P':
            n_params = 2*self.n_qubits*(self.n_layers+1)
        elif self.PQC_id == '7':
            n_params = (5*self.n_qubits-1)*self.n_layers
        elif self.PQC_id == '6':
            n_params = (self.n_qubits**2 + 3*self.n_qubits)*self.n_layers
        elif self.PQC_id == '3':
            n_params = (3*self.n_qubits-1)*self.n_layers
        elif self.PQC_id == 'generic':
            n_params = (6*self.n_qubits)*self.n_layers
        elif self.PQC_id == 'TTN':
            n_params = int(2**(np.log2(self.n_qubits)+1)-2 +1)
        elif self.PQC_id == 'MPS':
            n_params = 2*self.n_qubits - 1
        elif self.PQC_id == '10_local':
            n_params = self.n_qubits*self.n_layers+1
        elif self.PQC_id == '10_2des':
            n_params = self.n_qubits*(self.n_layers+1)
        elif self.PQC_id == '10_2':
            n_params = self.n_qubits*(self.n_layers+1)
        elif self.PQC_id == '10_3':
            n_params = self.n_qubits*(self.n_layers+1)
        elif self.PQC_id == '10_identity':
            n_params = self.n_qubits*(self.n_layers+1)*2
        else:
            raise ValueError('PQC weights not defined')
        return n_params


    def get_n_qubits(self):
        # set number of qubits and inputs
        if self.IEC_id in self.metadata['n_qubits_dict'].keys():
            n_qubits = self.metadata['n_qubits_dict'][self.IEC_id]
        else:
            n_qubits = self.n_inputs
        return n_qubits

    def get_measurements(self):
        # set number of measurements
        n_measurements_dict = {
            'measure_all': self.n_qubits,
            'measure_last': 1,
            #'none': 0,
            #'probs': self.n_qubits**2,
            #'samples': 1000*(self.n_qubits**2),
            }
        return n_measurements_dict[self.MC_id]







###############################################################################
class Rescale01(tf.keras.layers.Layer):
    def __init__(self, name='Rescale01'):
        super(Rescale01, self).__init__(name=name)

    def call(self, X):
        X = tf.divide(
                tf.subtract(
                    X, 
                    tf.reduce_min(X)
                ), 
                tf.subtract(
                    tf.reduce_max(X), 
                    tf.reduce_min(X)
                ),
            lambda: X
        )
        return X
###############################################################################
class EdgeNet(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_qubits, dp_noise, IEC_id, PQC_id, MC_id, repetitions, name='EdgeNet'):
        super(EdgeNet, self).__init__(name=name)

        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.repetitions = repetitions
                
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(IEC_id=IEC_id,
            PQC_id=PQC_id,
            MC_id=MC_id,
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01)
        
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])
        
        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and self.repetitions!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and self.repetitions==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

         # Classical readout layer
        self.readout_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self,X):
        '''forward pass of the edge network. '''

        # Constrcu the B matrix
        # bo = tf.matmul(Ro,X,transpose_a=True)
        # bi = tf.matmul(Ri,X,transpose_a=True)
        # Shape of B = N_edges x 6 (2x (3 + Hidden Dimension Size))
        # each row consists of two node that are connected in the input graph.
        # B  = tf.concat([bo, bi], axis=1) # n_edges x 6, 3-> r,phi,z 

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        # input_to_circuit = self.input_layer(B) * np.pi
        input_to_circuit = self.input_layer(X) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        
          
        # Get expectation values for all edges
        if self.repetitions==0:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data
            )
        else:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=self.repetitions
            )
    
        # Return the output of the final layer
        return self.readout_layer(exps)

class NodeNet(tf.keras.layers.Layer):
    def __init__(self, hid_dim, n_layers, n_qubits, dp_noise, IEC_id, PQC_id, MC_id, repetitions, name='NodeNet'):
        super(NodeNet, self).__init__(name=name)
        
        self.n_layers = n_layers
        self.n_qubits = n_qubits

               
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(
            IEC_id,
            PQC_id,
            MC_id,
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01
        )
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])

        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and  self.repetitions!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and  self.repetitions==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

        # Classical readout layer
        self.readout_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                hid_dim, 
                activation='relu'),
            Rescale01()
        ])

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self, X, e, Ri, Ro):
        '''forward pass of the node network. '''

        # The following lines constructs the M matrix
        # M matrix contains weighted averages of input and output nodes
        # the weights are the edge probablities.
        bo  = tf.matmul(Ro, X, transpose_a=True)
        bi  = tf.matmul(Ri, X, transpose_a=True) 
        Rwo = Ro * e[:,0]
        Rwi = Ri * e[:,0]
        mi = tf.matmul(Rwi, bo)
        mo = tf.matmul(Rwo, bi)
        # Shape of M = N_nodes x (3x (3 + Hidden Dimension Size))
        # mi: weighted average of input nodes
        # mo: weighted average of output nodes
        M = tf.concat([mi, mo, X], axis=1)

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(M) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        

        # Get expectation values for all nodes
        if self.repetitions==0:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data)
        else:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=self.repetitions)

        # Return the output of the final layer
        return self.readout_layer(exps)

###############################################################################
class GNN(tf.keras.Model):
    def __init__(self, parameters):
        ''' Init function of GNN, inits all GNN blocks. '''
        super(GNN, self).__init__(name='GNN')
        
        HID_DIM = parameters["HID_DIM"] if "HID_DIM" in parameters else None
        N_ITERS = parameters["N_ITERS"] if "N_ITERS" in parameters else None


        EN_N_QUBITS = parameters["EN_QC"]["N_QUBITS"] if "N_QUBITS" in parameters["EN_QC"] else None
        EN_PQC_ID = parameters["EN_QC"]["PQC_ID"] if "PQC_ID" in parameters["EN_QC"] else None
        EN_IEC_ID = parameters["EN_QC"]["IEC_ID"] if "IEC_ID" in parameters["EN_QC"] else None
        EN_MC_ID = parameters["EN_QC"]["MC_ID"] if "MC_ID" in parameters["EN_QC"] else None
        EN_N_LAYERS = parameters["EN_QC"]["N_LAYERS"] if "N_LAYERS" in parameters["EN_QC"] else None
        EN_REPETITIONS = parameters["EN_QC"]["REPETITIONS"] if "REPETITIONS" in parameters["EN_QC"] else None
        EN_DP_NOISE = parameters["EN_QC"]["DP_NOISE"] if "DP_NOISE" in parameters["EN_QC"] else None

        NN_N_QUBITS = parameters["NN_QC"]["N_QUBITS"] if "N_QUBITS" in parameters["NN_QC"] else None
        NN_PQC_ID = parameters["NN_QC"]["PQC_ID"] if "PQC_ID" in parameters["NN_QC"] else None
        NN_IEC_ID = parameters["NN_QC"]["IEC_ID"] if "IEC_ID" in parameters["NN_QC"] else None
        NN_MC_ID = parameters["NN_QC"]["MC_ID"] if "MC_ID" in parameters["NN_QC"] else None
        NN_N_LAYERS = parameters["NN_QC"]["N_LAYERS"] if "N_LAYERS" in parameters["NN_QC"] else None
        NN_REPETITIONS = parameters["NN_QC"]["REPETITIONS"] if "REPETITIONS" in parameters["NN_QC"] else None
        NN_DP_NOISE = parameters["NN_QC"]["DP_NOISE"] if "DP_NOISE" in parameters["NN_QC"] else None


        # Define Initial Input Layer
        self.InputNet =  tf.keras.layers.Dense(
            HID_DIM, input_shape=(3,),
            activation='relu',name='InputNet'
        )
        self.EdgeNet  = EdgeNet(EN_N_LAYERS, EN_N_QUBITS, EN_DP_NOISE, EN_IEC_ID, EN_PQC_ID, EN_MC_ID, EN_REPETITIONS, name='EdgeNet')
        self.NodeNet  = NodeNet(HID_DIM, NN_N_LAYERS, NN_N_QUBITS, NN_DP_NOISE, NN_IEC_ID, NN_PQC_ID, NN_MC_ID, NN_REPETITIONS, name='NodeNet')
        self.n_iters  = N_ITERS
    
    def call(self, graph_array):
        ''' forward pass of the GNN '''
        # decompose the graph array
        X, Ri, Ro = graph_array
        # execute InputNet to produce hidden dimensions
        H = self.InputNet(X)
        # add new dimensions to original X matrix
        H = tf.concat([H,X],axis=1)
        # recurrent iteration of the network
        for i in range(self.n_iters):
            e = self.EdgeNet(H, Ri, Ro)
            H = self.NodeNet(H, e, Ri, Ro)
            # update H with the output of NodeNet
            H = tf.concat([H,X],axis=1)
        # execute EdgeNet one more time to obtain edge predictions
        e = self.EdgeNet(H, Ri, Ro)
        # return edge prediction array
        return e



def train_qgnn(model_parameters, all_lca_shuffled, all_leaves_shuffled):
    # MODEL = parameters["MODEL"] if "MODEL" in parameters else None
    BATCH_SIZE = model_parameters["BATCH_SIZE"] if "BATCH_SIZE" in model_parameters else None
    OPTIMIZER = model_parameters["OPTIMIZER"] if "OPTIMIZER" in model_parameters else None
    LOSS_FUNC = model_parameters["LOSS_FUNC"] if "LOSS_FUNC" in model_parameters else None
    N_EPOCH = model_parameters["N_EPOCH"] if "N_EPOCH" in model_parameters else None
    LR_C = model_parameters["LR_C"] if "LR_C" in model_parameters else None
    N_TRAIN = model_parameters["N_TRAIN"] if "N_TRAIN" in model_parameters else None


    # Read config file
    # config = load_config(parse_args())
    # tools.config = config

    # Set GPU variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    # USE_GPU = (config['gpu']  != '-1')

    # Set number of thread to be used
    # os.environ['OMP_NUM_THREADS'] = str(config['n_thread'])  # set num workers
    # tf.config.threading.set_intra_op_parallelism_threads(config['n_thread'])
    # tf.config.threading.set_inter_op_parallelism_threads(config['n_thread'])

    # GNN.config = config
 
    # setup model
    model = GNN()

    # load data
    # train_data = get_dataset(config['train_dir'], N_TRAIN)
    train_list = [i for i in range(N_TRAIN)]

    # # execute the model on an example data to test things
    # X, Ri, Ro, y = train_data[0]
    # model([map2angle(X), Ri, Ro])

    # # print model summary
    # print(model.summary())

    # # Log initial parameters if new run
    # if config['run_type'] == 'new_run':    
    #     if config['log_verbosity']>=2:
    #         log_parameters(config['log_dir'], model.trainable_variables)
    epoch_start = 0

    #     # Test the validation and training set
    #     if config['n_valid']: test(config, model, 'valid')
    #     if N_TRAIN: test(config, model, 'train')
    # # Load old parameters if continuing run
    # elif config['run_type'] == 'continue':
    #     # load params 
    #     model, epoch_start = load_params(model, config['log_dir'])
    # else:
    #     raise ValueError('Run type not defined!')

    # Get loss function and optimizer
    loss_fn = getattr(tf.keras.losses, LOSS_FUNC)()
    opt = getattr(
        tf.keras.optimizers,
        OPTIMIZER)(learning_rate=LR_C
    )

    # # Print final message before training
    # if epoch_start == 0: 
    #     print(str(datetime.datetime.now()) + ': Training is starting!')
    # else:
    #     print(
    #         str(datetime.datetime.now()) 
    #         + ': Training is continuing from epoch {}!'.format(epoch_start+1)
    #         )

    # Start training
    for epoch in range(epoch_start, N_EPOCH):
        # shuffle(train_list) # shuffle the order every epoch

        for n_step in range(N_TRAIN//BATCH_SIZE):
            # start timer
            # t0 = datetime.datetime.now()  

            # iterate a step
            loss_eval, grads = batch_train_step(loss_fn, opt, model, n_step)
                        
            # end timer
            # dt = datetime.datetime.now() - t0  
            # t = dt.seconds + dt.microseconds * 1e-6 # time spent in seconds

            # # Print summary
            # print(
            #     str(datetime.datetime.now())
            #     + ": Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" \
            #     %(epoch+1, n_step+1, loss_eval.numpy() ,t / 60, t % 60)
            #     )
            
            # Start logging 
            
            # Log summary 
            # with open(config['log_dir']+'summary.csv', 'a') as f:
            #     f.write(
            #         '%d, %d, %f, %f\n' \
            #         %(epoch+1, n_step+1, loss_eval.numpy(), t)
            #         )

	       # Log parameters
        #     if config['log_verbosity']>=2:
        #         log_parameters(config['log_dir'], model.trainable_variables)

        #    # Log gradients
        #     if config['log_verbosity']>=2:
        #         log_gradients(config['log_dir'], grads)
            
        #     # Test every TEST_every
        #     if (n_step+1)%config['TEST_every']==0:
        #         test(config, model, 'valid')
        #         test(config, model, 'train')

    return model

def batch_train_step(model, loss_fn, opt, n_step, BATCH_SIZE):
    '''combines multiple  graph inputs and executes a step on their mean'''
    with tf.GradientTape() as tape:
        for batch in range(BATCH_SIZE):
            X, Ri, Ro, y = train_data[
                train_list[n_step*BATCH_SIZE+batch]
                ]

            label = tf.reshape(tf.convert_to_tensor(y),shape=(y.shape[0],1))
            
            if batch==0:
                # calculate weight for each edge to avoid class imbalance
                # weights = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                # weights = tf.reshape(tf.convert_to_tensor(weights),
                                    #  shape=(weights.shape[0],1))
                preds = model([map2angle(X),Ri,Ro])
                labels = label
            else:
                # weight = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                # weight = tf.reshape(tf.convert_to_tensor(weight),
                                    # shape=(weight.shape[0],1))

                # weights = tf.concat([weights, weight],axis=0)
                preds = tf.concat([preds, model([map2angle(X),Ri,Ro])],axis=0)
                labels = tf.concat([labels, label],axis=0)

        # loss_eval = loss_fn(labels, preds, sample_weight=weights)
        loss_eval = loss_fn(labels, preds)

    grads = tape.gradient(loss_eval, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss_eval, grads

# def true_fake_weights(labels):
#     ''' 
#     [weight of fake edges, weight of true edges]
    
#     weights are calculated using scripts/print_dataset_specs.py

#     '''
#     if tools.config['dataset'] == 'mu200':
#         weight_list = [1.102973565242351, 0.9146118742361756]
#     elif tools.config['dataset'] == 'mu200_1pT':
#         weight_list = [1.024985997012696, 0.9762031776515252]
#     elif tools.config['dataset'] == 'mu200_full':
#         weight_list = [0.5424779619482216, 6.385404773061769]
#     elif tools.config['dataset'] == 'mu10':
#         weight_list = [3.030203859885135, 0.5988062677334424]
#     elif tools.config['dataset'] == 'mu10_big':
#         weight_list = [0.9369978711656622, 1.0720851667609774]
#     else:
#         raise ValueError('dataset not defined')

#     return [weight_list[int(labels[i])] for i in range(labels.shape[0])]

def map2angle(arr0):
    # Mapping the cylindrical coordinates to [0,1]
    arr = np.zeros(arr0.shape, dtype=np.float32)
    r_min     = 0.
    r_max     = 1.1
    arr[:,0] = (arr0[:,0]-r_min)/(r_max-r_min)    



    # if (tools.config['dataset'] == 'mu200') or (tools.config['dataset'] == 'mu200_full'):
    phi_min   = -1.0
    phi_max   = 1.0
    arr[:,1]  = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
    z_min     = 0
    z_max     = 1.1
    arr[:,2]  = (np.abs(arr0[:,2])-z_min)/(z_max-z_min)  # take abs of z due to symmetry of z

    # elif tools.config['dataset'] == 'mu200_1pT':
    #     phi_max  = 1.
    #     phi_min  = -phi_max
    #     z_max    = 1.1
    #     z_min    = -z_max
    #     arr[:,1] = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
    #     arr[:,2] = (arr0[:,2]-z_min)/(z_max-z_min) 

    # elif (tools.config['dataset'] == 'mu10') or (tools.config['dataset'] == 'mu10_big'):
    #     phi_min   = -1.0
    #     phi_max   = 1.0
    #     arr[:,1] = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
    #     z_min     = -1.1
    #     z_max     = 1.1
    #     arr[:,2] = (arr0[:,2]-z_min)/(z_max-z_min) 

    # mapping_check(arr)
    return arr

def mapping_check(arr):
# check if every element of the input array is within limits [0,2*pi]
    for row in arr:
        for item in row:
            if (item > 1) or (item < 0):
                raise ValueError('WARNING!: WRONG MAPPING!!!!!!')

# -----------------

# class Generator(TorchConnector):
#     def __init__(self,
#                     latentDim:int=100, 
#                     outputShape:Tuple(int,int,int)=(1,4,4)
#     ) -> None:
#         num_inputs = outputShape[1]*outputShape[2]
#         feature_map = ZZFeatureMap(num_inputs)
#         ansatz = RealAmplitudes(num_inputs, entanglement="linear", reps=1)

#         # Define quantum circuit of num_qubits = input dim
#         # Append feature map and ansatz
#         qc = QuantumCircuit(num_inputs)
#         qc.append(feature_map, range(num_inputs))
#         qc.append(ansatz, range(num_inputs))


#         # Define CircuitQNN and initial setup
#         parity = lambda x: "{:b}".format(x).count("1") % 2  # optional interpret function
#         output_shape = 2  # parity = 0, 1
#         qnn = CircuitQNN(
#             qc,
#             input_params=feature_map.parameters,
#             weight_params=ansatz.parameters,
#             interpret=parity,
#             output_shape=output_shape,
#         )

# # Set up PyTorch module
# # Reminder: If we don't explicitly declare the initial weights
# # they are chosen uniformly at random from [-1, 1].
# # initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn2.num_weights) - 1)

#     #     super(Generator, self).__init__(self.model, initialWeights)

#     # def forward(self, z):
#     #     graph = self.model(z)
#     #     graph = graph.view(graph.size(0), *self.outputShape)
#     #     return graph

# class Discriminator(nn.Module):
#     def __init__(self, 
#                     inputShape:Tuple(int,int,int)=(1,4,4)
#     ) -> None:
#         super(Discriminator, self).__init__()

#         self.inputShape = inputShape

#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(self.inputShape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)

#         return validity

# def train_discriminator(
#     generator_input: Dict
# ) -> nn.Module:
#     pass

# def train_generator(
#     generator_input: Tuple[List, List]
# ) -> nn.Module:
#     pass