import torch
from .instructors.XNRI_enc import XNRIENCIns
from .models.encoder import GNNENC
from .models.decoder import GNNDEC
from .models.nri import NRIModel
from torch.nn.parallel import DataParallel
# from generate.load import load_nri
from itertools import permutations
import numpy as np
from torch import LongTensor, FloatTensor

from torch.utils.data.dataset import TensorDataset


def train_qgnn(model_parameters, all_leaves_shuffled, all_lca_shuffled):
    # load data
    EDGE_TYPE = model_parameters["EDGE_TYPE"] if "EDGE_TYPE" in model_parameters else None
    SIZE = model_parameters["SIZE"] if "SIZE" in model_parameters else None
    REDUCE = model_parameters["REDUCE"] if "REDUCE" in model_parameters else None
    N_HID = model_parameters["N_HID"] if "N_HID" in model_parameters else None
    DIM = model_parameters["DIM"] if "DIM" in model_parameters else None


    # data, es, _ = load_nri(all_leaves_shuffled, num_of_leaves)
    # generate edge list of a fully connected graph
    es = LongTensor(np.array(list(permutations(range(SIZE), 2))).T)

    # dim = 4

    modes = all_lca_shuffled.keys()
    data = dict()
    for mode in modes:
        x_data = []
        y_data = []
        for topology_it in range(len(all_lca_shuffled[mode])):
            for i in range(len(all_lca_shuffled[mode][topology_it])):
                x_data.append(all_leaves_shuffled[mode][topology_it][i])
                y_data.append(all_lca_shuffled[mode][topology_it][i])
        data[mode] = (LongTensor(y_data), FloatTensor(x_data))

    EDGE_TYPE = int(np.array(y_data).max())+1 # get the num of childs from the label list

    encoder = GNNENC(DIM, N_HID, EDGE_TYPE, reducer=REDUCE)
    model = NRIModel(encoder, es, SIZE)
    model = DataParallel(model)
    ins = XNRIENCIns(model_parameters, model, data, es, model_parameters)
    ins.train()