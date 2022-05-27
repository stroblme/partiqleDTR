import torch
from instructors.XNRI_enc import XNRIENCIns
from argparse import ArgumentParser
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from torch.nn.parallel import DataParallel
from generate.load import load_nri


def train_qgnn(model_parameters, all_leaves_shuffled, all_lca_shuffled):
    # load data
    EDGE_TYPE = model_parameters["EDGE_TYPE"] if "EDGE_TYPE" in model_parameters else None
    SIZE = model_parameters["SIZE"] if "SIZE" in model_parameters else None
    REDUCE = model_parameters["REDUCE"] if "REDUCE" in model_parameters else None
    SKIP = model_parameters["SKIP"] if "SKIP" in model_parameters else None
    N_HID = model_parameters["N_HID"] if "N_HID" in model_parameters else None


    num_of_leaves = all_lca_shuffled["train"][0][0].shape[0]

    data, es, _ = load_nri(all_leaves_shuffled, num_of_leaves)
    dim = 4
    encs = {
        'GNNENC': GNNENC,
        'RNNENC': RNNENC,
        'AttENC': AttENC,
    }
    decs = {
        'GNNDEC': GNNDEC,
        'RNNDEC': RNNDEC,
        'AttDEC': AttDEC,
    }
    encoder = GNNENC(dim, N_HID, EDGE_TYPE, reducer=REDUCE)
    decoder = GNNDEC(dim, EDGE_TYPE, N_HID, N_HID, N_HID, skip_first=SKIP)
    model = NRIModel(encoder, decoder, es, SIZE)
    model = DataParallel(model)
    ins = XNRIENCIns(model, data, es, model_parameters)
    ins.train()