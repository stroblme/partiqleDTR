import numpy as np
import matplotlib.pyplot as plt

import git

import torch as t
from torch.nn.parallel import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import mlflow

from .nri_gnn import bb_NRIModel, rel_pad_collate_fn
from .graph_visualization import GraphVisualization

from typing import Dict

import logging
log = logging.getLogger(__name__)


def train_qgnn( torch_dataset_lca_and_leaves:Dict, n_momenta:int,
                n_blocks:int, dim_feedforward:int, n_layers_mlp:int, n_additional_mlp_layers:int, n_final_mlp_layers:int,
                dropout_rate:float, learning_rate:float, learning_rate_decay:int, gamma:float, batch_size:int, epochs:int):
    n_fsps = int(max([len(subset[0]) for _, subset in torch_dataset_lca_and_leaves.items()]))+1

    model = bb_NRIModel(infeatures=n_momenta,
                        num_classes=n_fsps,
                        n_blocks=n_blocks,
                        dim_feedforward=dim_feedforward,
                        n_layers_mlp=n_layers_mlp,
                        n_additional_mlp_layers=n_additional_mlp_layers,
                        n_final_mlp_layers=n_final_mlp_layers,
                        dropout=dropout_rate,
                        factor=True,
                        tokenize=None,
                        embedding_dims=None,
                        batchnorm=True,
                        symmetrize=True)
    model = DataParallel(model)
    ins = Instructor(model, torch_dataset_lca_and_leaves, learning_rate, learning_rate_decay, gamma, batch_size, epochs)
    

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_tag("git_hash", str(sha))

    return ins.train()


class DataWrapper(Dataset):
    """
    A wrapper for t.utils.data.Dataset.
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
    def __init__(self, model: DataParallel, data: dict,
                learning_rate: float, learning_rate_decay: int, gamma: float, batch_size:int, epochs:int):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        self.model = model
        
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = t.optim.Adam(self.model.parameters(), lr=learning_rate)
        # learning rate scheduler, same as in NRI
        self.scheduler = StepLR(self.opt, step_size=learning_rate_decay, gamma=gamma)


    def train(self):
        log.info(f'Training started with a batch size of {self.batch_size}')
        result = None            
        best_acc = 0
        for epoch in range(1, 1 + self.epochs):
            for mode in ["train", "val"]:
                data_batch = DataLoader(
                                        DataWrapper(self.data[mode]),
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        collate_fn=rel_pad_collate_fn) # to handle varying input size

                epoch_loss = 0.
                epoch_acc = 0.

                log.info(f"Running epoch {epoch} in mode {mode}")
                for states, labels in data_batch:
                    scale = 1 / labels.size(1) # get the scaling dependend on the number of classes

                    if mode == "train":
                        self.model.train() # set the module in training mode

                        prob = self.model.module(states)
                        loss = cross_entropy(prob, labels, ignore_index=-1)
                        acc = self.edge_accuracy(prob, labels)

                        # do the actual optimization
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        if labels.numpy().min() < -1:
                            raise Exception(f"Found graph with negative values: {labels.numpy()}")
                    elif mode == "val":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            prob = self.model.module(states)

                            loss = cross_entropy(prob, labels, ignore_index=-1)
                            acc = self.edge_accuracy(prob, labels)
                    elif mode == "test":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            prob = self.model.module(states)

                            loss = cross_entropy(prob, labels, ignore_index=-1)
                            acc = self.edge_accuracy(prob, labels)
                    else:
                        log.error("Unknown mode")

                    epoch_loss += scale * loss
                    epoch_acc += scale * acc

                    if acc > best_acc and mode == "val":
                        # update the current best model when approaching a higher accuray
                        best_acc = acc
                        result = self.model
                        try:
                            c_plt = self.plotBatchGraphs(prob, labels)
                        except Exception as e:
                            log.error(f"Exception occured when trying to plot graphs: {e}\n\tThe lcag matrices were:\n\t{labels.numpy()}\n\tand\n\t{prob.numpy()}")

                        mlflow.log_figure(c_plt.gcf(), f"e{epoch}_sample_graph.png")

                epoch_loss /= len(data_batch) # to the already scaled loss, apply the batch size scaling
                epoch_acc /= len(data_batch) # to the already scaled accuracy, apply the batch size scaling

                mlflow.log_metric(key=f"{mode}_accuracy", value=epoch_acc.item(), step=epoch)
                mlflow.log_metric(key=f"{mode}_loss", value=epoch_loss.item(), step=epoch)

                # learning rate scheduling
                self.scheduler.step()

    

        return {
            "model_qgnn":result
        }

    def plotBatchGraphs(self, batch_logits, batch_ref, rows=4, cols=2):
        fig, ax = plt.subplots(rows, cols, figsize=(15,15), gridspec_kw={'width_ratios': [1, 1]})
        fig.tight_layout()
        it = 0
        for logits, lcag_ref in zip(batch_logits, batch_ref):
            lcag = logits.max(0)[1]
            graph = GraphVisualization()

            graph.lca2graph(lcag)
            plt.sca(ax[it][0])
            graph.visualize(opt="max", ax=ax[it][0])

            graph_ref = GraphVisualization()

            graph_ref.lca2graph(lcag_ref)
            plt.sca(ax[it][1])
            graph_ref.visualize(opt="max", ax=ax[it][1])

            if it*cols>rows:
                break

            it += 1

        return plt

    def edge_accuracy(self, logits:t.Tensor, labels:t.Tensor)->float:
        # logits: [Batch, Classes, LCA_0, LCA_1]
        probs = logits.softmax(1) # get softmax for probabilities
        preds = probs.max(1)[1] # find maximum across the classes
        correct = (labels==preds).sum().float()
        return correct/(labels.size(1)*labels.size(2))           

