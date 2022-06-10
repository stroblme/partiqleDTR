import torch
from .models.encoder import GNNENC
from .models.nri import NRIModel
from torch.nn.parallel import DataParallel
# from generate.load import load_nri
from itertools import permutations
import numpy as np
from torch import LongTensor, FloatTensor

from torch.utils.data.dataset import TensorDataset

import torch
from torch import Tensor
# import config as cfg
from torch.nn.functional import cross_entropy
from .utils.torch_extension import edge_accuracy, asym_rate
from torch.utils.data.dataset import TensorDataset
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np

import logging
log = logging.getLogger(__name__)

def train_qgnn(model_parameters, torch_dataset_lca_and_leaves):
    # load data
    # SIZE = model_parameters["SIZE"] if "SIZE" in model_parameters else None
    REDUCE = model_parameters["REDUCE"] if "REDUCE" in model_parameters else None
    N_HID = model_parameters["N_HID"] if "N_HID" in model_parameters else None
    DIM = model_parameters["DIM"] if "DIM" in model_parameters else None


    # data, es, _ = load_nri(all_leaves_shuffled, num_of_leaves)
    # generate edge list of a fully connected graph

    EDGE_TYPE = int(max([np.array(subset[0]).max() for _, subset in torch_dataset_lca_and_leaves.items()]))+1 # get the num of childs from the label list
    SIZE = int(max([np.array(subset[0]).shape[1] for _, subset in torch_dataset_lca_and_leaves.items()]))

    # es = LongTensor(np.array(list(permutations(range(SIZE), 2))).T)
    es = list(permutations(range(SIZE), 2))
    es.sort(key=lambda es: sum(es))
    es = es[1::2]
    es = LongTensor(np.array(es).T)

    encoder = GNNENC(DIM, N_HID, EDGE_TYPE, reducer=REDUCE)
    model = NRIModel(encoder, es, SIZE)
    model = DataParallel(model)
    ins = Instructor(model_parameters, model, torch_dataset_lca_and_leaves, es)
    ins.train()



class DataWrapper(Dataset):
    """
    A wrapper for torch.utils.data.Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Instructor():
    """
    Train the encoder in an supervised manner given the ground truth relations.
    """
    def __init__(self, model_parameters, model: torch.nn.DataParallel, data: dict,  es: np.ndarray):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        LR = model_parameters["LR"] if "LR" in model_parameters else None
        LR_DECAY = model_parameters["LR_DECAY"] if "LR_DECAY" in model_parameters else None
        GAMMA = model_parameters["GAMMA"] if "GAMMA" in model_parameters else None
        SIZE = model_parameters["SIZE"] if "SIZE" in model_parameters else None
        BATCH_SIZE = model_parameters["BATCH_SIZE"] if "BATCH_SIZE" in model_parameters else None
        EPOCHS = model_parameters["EPOCHS"] if "EPOCHS" in model_parameters else None


        # super(XNRIENCIns, self).__init__(cmd)
        self.model = model
        
        self.data = {key: TensorDataset(*value)
                     for key, value in data.items()}
        # self.data = data
        self.es = torch.LongTensor(es)
        # number of nodes
        self.size = SIZE
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE
        # optimizer
        self.opt = optim.Adam(self.model.parameters(), lr=LR)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=LR_DECAY, gamma=GAMMA)

    @staticmethod
    def optimize(opt: Optimizer, loss: torch.Tensor):
        """
        Optimize the parameters based on the loss and the optimizer.

        Args:
            opt: optimizer
            loss: loss, a scalar
        """
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(inputs, batch_size: int, shuffle: bool=True):
        """
        Return a dataloader given the input and the batch size.
        """
        data = DataWrapper(inputs)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle)
        return batches

    def train(self):
        # use the accuracy as the metric for model selection, default: 0
        # val_best = 0
        # path to save the current best model
        # prefix = '/'.join(cfg.log.split('/')[:-1])
        # name = '{}/best.pth'.format(prefix)train_steps
        for epoch in range(1, 1 + self.epochs):
            self.model.train()
            # shuffle the data at each epoch
            data = self.load_data(self.data['train'], self.batch_size)
            loss_a = 0.
            N = 0.
            log.info(f'Training started with a batch size of {self.batch_size}')
            for adj, states in data:
                # if cfg.gpu:
                #     adj = adj.cuda()
                #     states = states.cuda()
                scale = len(states) / self.batch_size
                # N: number of samples, equal to the batch size with possible exception for the last batch
                N += scale
                loss_a += scale * self.train_nri(states, adj)
            loss_a /= N 
            log.info(f'Epoch {epoch:03d} finished with an average loss of {loss_a:.3e}')
            # acc = self.report('val')

            # val_cur = max(acc, 1 - acc)
            # if val_cur > val_best:
                # update the current best model when approaching a higher accuray
                # val_best = val_cur
                # torch.save(self.model.module.state_dict(), name)

            self.scheduler.step()
        # learning rate scheduling
        # if self.cmd.epochs > 0:
            # self.model.module.load_state_dict(torch.load(name))
        # _ = self.report('test')

        return self.model

    def report(self, name: str) -> float:
        """
        Evaluate the accuracy.

        Args:
            name: 'train' / 'val' / 'test'
        
        Return:
            acc: accuracy of relation reconstruction
        """
        loss, acc, rate, sparse = self.evaluate(self.data[name])
        log.info('{} acc {:.4f} _acc {:.4f} rate {:.4f} sparse {:.4f}'.format(
            name, acc, 1 - acc, rate, sparse))
        return acc

    def train_nri(self, states: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            states: [batch, step, node, dim], observed node states
            adj: [batch, E, K], ground truth interacting relations

        Return:
            loss: cross-entropy of edge classification
        """
        prob = self.model.module.predict_relations(states)

        adj_ut = []
        for batch in range(adj.shape[0]):
            adj_ut_batch = []
            for row in range(adj.shape[1]):
                for col in range(adj.shape[2]):
                    if row < col:
                        # adj_ut_batch.append(adj[batch][row][col])
                        adj_ut.append(adj[batch][row][col])
            # adj_ut.append(adj_ut_batch)
        adj_ut = LongTensor(adj_ut)

        # loss = cross_entropy(prob.view(-1, prob.shape[-1]), adj.transpose(0, 1).flatten())
        loss = cross_entropy(prob.view(-1, prob.shape[-1]), adj_ut)
        
        self.optimize(self.opt, loss)
        return loss

    def evaluate(self, test):
        """
        Evaluate related metrics to monitor the training process.

        Args:
            test: data set to be evaluted

        Return:
            loss: loss_nll + loss_kl (+ loss_reg) 
            acc: accuracy of relation reconstruction
            rate: rate of assymmetry
            sparse: rate of sparsity in terms of the first type of edge
        """
        acc, rate, sparse, losses = [], [], [], []
        data = self.load_data(test, self.batch_size)
        N = 0.
        with torch.no_grad():
            for adj, states in data:
                print(states.shape)
                prob = self.model.module.predict_relations(states)

                scale = len(states) / self.batch_size
                N += scale

                # use loss as the validation metric
                loss = cross_entropy(prob.view(-1, prob.shape[-1]), adj.transpose(0, 1).flatten())
                # scale all metrics to match the batch size
                loss = loss * scale
                losses.append(loss)

                acc.append(scale * edge_accuracy(prob, adj))
                _, p = prob.max(-1)
                rate.append(scale * asym_rate(p.t(), self.size))
                sparse.append(prob.max(-1)[1].float().mean() * scale)
        loss = sum(losses) / N
        acc = sum(acc) / N
        rate = sum(rate) / N
        sparse = sum(sparse) / N
        return loss, acc, rate, sparse
