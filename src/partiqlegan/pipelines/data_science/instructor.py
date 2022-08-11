import time

import matplotlib.pyplot as plt

import torch as t
from torch.nn.parallel import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import mlflow

from .utils import rel_pad_collate_fn
from .graph_visualization import GraphVisualization

from typing import Dict

import logging
log = logging.getLogger(__name__)

class DataWrapper(Dataset):
    """
    A wrapper for t.utils.data.Dataset.
    """
    def __init__(self, data, normalize=False):
        dmax = 0
        dmin = 1
        self.data = data
        if normalize:
            for i, event in enumerate(data.x):
                dmax = event.max() if event.max() > dmax else dmax
                dmin = event.min() if event.min() < dmin else dmin
            for i, event in enumerate(data.x):
                self.data.x[i] = (event-dmin)/(dmax-dmin)
            dmax = 0
            dmin = 1
            for i, event in enumerate(data.x):
                dmax = event.max() if event.max() > dmax else dmax
                dmin = event.min() if event.min() < dmin else dmin
            assert dmax==1 and dmin==0
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Instructor():
    """
    Train the encoder in an supervised manner given the ground truth relations.
    """
    def __init__(self, model: DataParallel, data: dict,
                learning_rate: float, learning_rate_decay: int, gamma: float, batch_size:int, epochs:int, normalize: bool,
                plot_mode:str="val"):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """
        self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        self.device = 'cpu'

        self.model = model
        self.model.module.to(self.device)

        for p in self.model.parameters():
            p.register_hook(lambda grad: t.clamp(grad, -1000, 1000))
        
        self.pytorch_total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("Total trainable parameters", self.pytorch_total_params)
        
        self.plot_mode = plot_mode
        self.data = data
        self.epochs = epochs
        self.normalize = normalize
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
                                        DataWrapper(self.data[mode], normalize=self.normalize),
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        collate_fn=rel_pad_collate_fn) # to handle varying input size

                epoch_loss = 0.
                epoch_acc = 0.
                epoch_grad = []

                log.info(f"Running epoch {epoch} in mode {mode} over {len(data_batch)} samples")
                for states, labels in data_batch:
                    start = time.time()
                    states = [s.to(self.device) for s in states]
                    labels = labels.to(self.device)
                    gradients = t.zeros(len([p for p in self.model.parameters()]))
                    scale = 1 / labels.size(1) # get the scaling dependend on the number of classes

                    if mode == "train":
                        self.model.train() # set the module in training mode

                        logits = self.model.module(states)
                        loss = cross_entropy(logits, labels, ignore_index=-1)
                        acc = self.edge_accuracy(logits, labels)

                        # do the actual optimization
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                        gradients += t.Tensor([p.grad.norm() for p in self.model.parameters()])
                        log.info(f"Graients: {gradients}")

                        labels = labels.cpu()
                        if labels.numpy().min() < -1:
                            raise Exception(f"Found graph with negative values: {labels.numpy()}")
                    elif mode == "val":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            logits = self.model.module(states)

                            loss = cross_entropy(logits, labels, ignore_index=-1)
                            acc = self.edge_accuracy(logits, labels)
                    elif mode == "test":
                        self.model.module.eval() # trigger evaluation forward mode
                        with t.no_grad(): # disable autograd in tensors

                            logits = self.model.module(states)

                            loss = cross_entropy(logits, labels, ignore_index=-1)
                            acc = self.edge_accuracy(logits, labels)
                    else:
                        log.error("Unknown mode")

                    epoch_loss += scale * loss
                    epoch_acc += scale * acc
                    epoch_grad.append(scale * gradients)

                    if acc > best_acc and mode == self.plot_mode:
                        # update the current best model when approaching a higher accuray
                        best_acc = acc
                        result = self.model
                        try:
                            c_plt = self.plotBatchGraphs(logits.cpu(), labels)
                            mlflow.log_figure(c_plt.gcf(), f"{mode}_e{epoch}_sample_graph.png")
                        except Exception as e:
                            log.error(f"Exception occured when trying to plot graphs: {e}\n\tThe lcag matrices were:\n\t{labels.numpy()}\n\tand\n\t{logits.cpu().detach().numpy()}")


                    log.info(f"Sample evaluation took {time.time() - start} seconds. Loss was {scale*loss.item()}")

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

