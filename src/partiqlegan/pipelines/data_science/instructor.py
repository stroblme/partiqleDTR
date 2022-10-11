import time
import traceback
import random

import matplotlib.pyplot as plt

import torch as t
from torch.nn.parallel import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import numpy as np

import mlflow

import torchinfo

from .utils import rel_pad_collate_fn
from .graph_visualization import GraphVisualization
from .gradients_visualization import heatmap, annotate_heatmap

from typing import Dict

import logging

log = logging.getLogger(__name__)


class GradientsNanException(RuntimeError):
    pass


class DataWrapper(Dataset):
    """
    A wrapper for t.utils.data.Dataset.
    """

    def __init__(self, data, normalize=False, normalize_individually=True, zero_mean=False):
        dmax = 0
        dmin = 1

        # normalize_individually = True
        # zero_mean = False

        self.data = data
        if normalize=="one":
            if not normalize_individually:
                for i, event in enumerate(data.x):
                    dmax = event.max() if event.max() > dmax else dmax
                    dmin = event.min() if event.min() < dmin else dmin
                # for i, event in enumerate(data.x):
                #     self.data.x[i] = (event-dmin)/(dmax-dmin)
                # dmax = 0
                # dmin = 1
                # for i, event in enumerate(data.x):
                #     dmax = event.max() if event.max() > dmax else dmax
                #     dmin = event.min() if event.min() < dmin else dmin
                # assert dmax==1 and dmin==0
            for i, event in enumerate(data.x):
                if normalize_individually:
                    dmax = event.max()
                    dmin = event.min()

                if zero_mean:
                    self.data.x[i] = (event - (dmax - dmin)/2) / (dmax - dmin)
                else:
                    # self.data.x[i] = (event - dmin) / (dmax - dmin)
                    self.data.x[i] = (event) / (dmax - dmin)

        elif normalize=="zmuv":
            for i, event in enumerate(data.x):
                # adj_mean = t.repeat_interleave(t.Tensor([data.mean]), event.shape[0], dim=0)
                # adj_std = t.repeat_interleave(t.Tensor([data.std]), event.shape[0], dim=0)
                adj_mean = np.repeat([data.mean], event.shape[0], axis=0)
                adj_std = np.repeat([data.std], event.shape[0], axis=0)
                self.data.x[i] = (event - adj_mean)/adj_std

        # fill diagonal with zeros to ignore in loss
        for i, lcag in enumerate(data.y):
            np.fill_diagonal(self.data.y[i], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def calculate_class_weights(dataloader, num_classes, num_batches=100, amp_enabled=False):
    """ Calculates class weights based on num_batches of the dataloader

    This assumes there exists a -1 padding value that is not part of the class weights.
    Any classes not found will have a weight of one set

    Args:
        dataloader(torch.Dataloader): Dataloader to iterate through when collecting batches
        num_classes(int): Number of classes
        num_batches(int, optional): Number of batches from dataloader to use to approximate class weights
        amp_enabled(bool, optional): Enabled mixed precision. Creates weights tensor as half precision
    Return:
        (torch.tensor): Tensor of class weights, normalised to 1
    """
    weights = t.zeros((num_classes,))
    for i, batch in zip(range(num_batches), dataloader):
        index, count = t.unique(batch[1], sorted=True, return_counts=True)
        # TODO: add padding value as input to specifically ignore
        if -1 in index:
            # This line here assumes that the lowest class found is -1 (padding) which should be ignored
            weights[index[1:]] += count[1:]
        else:
            weights[index] += count

    # The weights need to be the invers, since we scale down the most common classes
    weights = 1 / weights
    # Set inf to 1
    weights = t.nan_to_num(weights, posinf=float('nan'))
    # And normalise to sum to 1
    weights = weights / weights.nansum()
    # Finally, assign default value to any that were missing during calculation time
    weights = t.nan_to_num(weights, nan=1)

    return weights

class Instructor:
    """
    Train the encoder in an supervised manner given the ground truth relations.
    """

    def __init__(
        self,
        model: DataParallel,
        data: dict,
        learning_rate: float,
        learning_rate_decay: int,
        gamma: float,
        batch_size: int,
        epochs: int,
        normalize: bool,
        normalize_individually: bool,
        zero_mean: bool,
        plot_mode: str = "val",
        plotting_rows: int = 4,
        log_gradients:bool = False,
        detectAnomaly: bool = False,
        device: str = "cpu",
        n_fsps=-1,
        n_classes=-1,
        gradients_clamp=1000,
        gradients_spreader=1e-10,
        model_state_dict=None,
        optimizer_state_dict=None,
    ):
        """
        Args:
            model: an auto-encoder
            data: train / val /test set
            es: edge list
            cmd: command line parameters
        """

        self.device = t.device(
            "cuda" if t.cuda.is_available() and device != "cpu" else "cpu"
        )

        self.model = model
        # self.model.to(self.device)
        log.info(f"Testing model..")
        mlflow.log_text(
            str(
                torchinfo.summary(
                    model, input_size=(n_fsps, batch_size, 4), device=self.device
                )
            ),
            "model_printout.txt",
        )
        for p in self.model.parameters():
            p.register_hook(
                lambda grad: t.clamp(grad, -gradients_clamp, gradients_clamp)
            )
            p.register_hook(
                lambda grad: t.where(
                    grad < gradients_spreader,
                    t.rand(1) * gradients_spreader * 1e1,
                    grad,
                )
            )

        self.pytorch_total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("Total trainable parameters", self.pytorch_total_params)

        self.plot_mode = plot_mode
        self.plotting_rows = plotting_rows  
        self.log_gradients = log_gradients
        self.data = data
        self.epochs = epochs
        self.normalize = normalize
        self.normalize_individually = normalize_individually
        self.zero_mean = zero_mean
        self.batch_size = batch_size
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=False)

        self.n_classes = n_classes
        
        # learning rate scheduler, same as in NRI

        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.scheduler = StepLR(
            self.optimizer, step_size=learning_rate_decay, gamma=gamma
        )
        self.detectAnomaly = detectAnomaly  # TODO: introduce as parameter if helpful

    def train(self, start_epoch=1):
        if self.detectAnomaly:
            log.info(f"Anomaly detection enabled")
            t.autograd.set_detect_anomaly(True)

        log.info(f"Training started with a batch size of {self.batch_size}")
        result = None
        best_acc = 0
        all_grads = []
        checkpoint = None

        try:  # catch things like gradient nan exceptions
            for epoch in range(start_epoch, 1 + self.epochs):
                logits_for_plotting = []
                labels_for_plotting = []
                for mode in ["train", "val"]:
                    data_batch = DataLoader(
                        DataWrapper(self.data[mode], normalize=self.normalize, normalize_individually=self.normalize_individually, zero_mean=self.zero_mean),
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=rel_pad_collate_fn,
                    )  # to handle varying input size

                    weights = calculate_class_weights(data_batch, self.n_classes, len(data_batch), False)
                    weights = weights.to(self.device)

                    epoch_loss = 0.0
                    epoch_acc = 0.0

                    if self.log_gradients:
                        epoch_grad = t.zeros(len([p for p in self.model.parameters()][0]))

                    # for i in range(10):
                    #     all_grads.append(epoch_grad+i)
                    # g_plt=self.plotGradients(all_grads)
                    # mlflow.log_figure(g_plt.gcf(), f"gradients.png")
                    log.info(
                        f"Running epoch {epoch} in mode {mode} over {len(data_batch)*self.batch_size} samples"
                    )
                    for i, (states, labels) in enumerate(data_batch):
                        sample_start = time.time()
                        states = [s.to(self.device) for s in states]
                        labels = labels.to(self.device)
                        scale = 1 / labels.size(
                            1
                        )  # get the scaling dependent on the number of classes

                        if mode == "train":
                            self.model.train()  # set the module in training mode

                            logits = self.model(states)
                            loss = cross_entropy(logits, labels, weight=weights, ignore_index=-1)
                            acc = self.edge_accuracy(logits, labels)

                            # self.plotBatchGraphs(logits, labels)
                            
                            # do the actual optimization
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            if self.log_gradients:
                                # raise Exception("test")
                                epoch_grad += t.Tensor(
                                    [p.grad for p in self.model.parameters()][0]
                                )  # only log the first layer (vqc)
                                if t.all(t.isnan(epoch_grad)):
                                    log.error(
                                        f"All gradients became nan in epoch {epoch} after iteration {i}.\nInput was\n{states}.\nPredicted was\n{logits}.\nGradients are\n{epoch_grad}"
                                    )
                                    raise GradientsNanException
                                elif t.any(
                                    t.isnan(epoch_grad)
                                ):  # TODO: we are checking for "all" instead of "any" since there were cases where the gradients became nan but training successfully continued (investigate in samples!)
                                    log.error(
                                        f"At least one gradient became nan in epoch {epoch} after iteration {i}.\nInput was\n{states}.\nPredicted was\n{logits}.\nGradients are\n{epoch_grad}"
                                    )
                                else:
                                    log.debug(
                                        f"Gradients in epoch {epoch}, iteration {i}: {epoch_grad}"
                                    )

                            # for i in range(self.epochs):
                            #     all_grads.append(scale * epoch_grad)
                            # g_plt = self.plotGradients(all_grads, figsize=(16,12))
                            # mlflow.log_figure(g_plt.gcf(), f"gradients.png")

                            labels = labels.cpu()
                            if labels.numpy().min() < -1:
                                raise Exception(
                                    f"Found graph with negative values: {labels.numpy()}"
                                )
                        elif mode == "val":
                            self.model.eval()  # trigger evaluation forward mode
                            with t.no_grad():  # disable autograd in tensors

                                logits = self.model(states)

                                loss = cross_entropy(logits, labels, weight=weights, ignore_index=-1)
                                acc = self.edge_accuracy(logits, labels)
                        elif mode == "test":
                            self.model.eval()  # trigger evaluation forward mode
                            with t.no_grad():  # disable autograd in tensors

                                logits = self.model(states)

                                loss = cross_entropy(logits, labels, weight=weights, ignore_index=-1)
                                acc = self.edge_accuracy(logits, labels)
                        else:
                            log.error("Unknown mode")

                        epoch_loss += scale * loss.item()
                        epoch_acc += scale * acc.item()

                        if mode == "train":
                            log.debug(
                                f"Sample evaluation in epoch {epoch}, iteration {i} took {time.time() - sample_start} seconds. Loss was {scale*loss.item()}"
                            )

                        if mode == self.plot_mode:
                            logits_for_plotting.extend(
                                [*logits]
                            )  # flatten along the batch size
                            labels_for_plotting.extend(
                                [*labels]
                            )  # flatten along the batch size
                        # if len(logits_for_plotting) > self.plotting_rows:
                        #     selected_logits = [random.choice(logits_for_plotting).cpu() for i in range(self.plotting_rows)]
                        #     selected_labels = [random.choice(labels_for_plotting) for i in range(self.plotting_rows)]

                        #     c_plt = self.plotBatchGraphs(selected_logits, selected_labels, rows=self.plotting_rows, cols=2)

                    epoch_loss /= len(
                        data_batch
                    )  # to the already scaled loss, apply the number of all iterations (no. of mini batches)
                    epoch_acc /= len(
                        data_batch
                    )  # to the already scaled accuracy, apply the number of all iterations (no. of mini batches)

                    if epoch_acc > best_acc and mode == self.plot_mode:
                        # update the current best model when approaching a higher accuray
                        best_acc = epoch_acc
                        result = self.model
                        try:
                            selected_logits = [
                                random.choice(logits_for_plotting).cpu()
                                for i in range(self.plotting_rows)
                            ]
                            selected_labels = [
                                random.choice(labels_for_plotting)
                                for i in range(self.plotting_rows)
                            ]

                            c_plt = self.plotBatchGraphs(
                                selected_logits,
                                selected_labels,
                                rows=self.plotting_rows,
                                cols=2,
                            )
                            mlflow.log_figure(
                                c_plt.gcf(), f"{mode}_e{epoch}_sample_graph.png"
                            )
                        except Exception as e:
                            log.error(
                                f"Exception occured when trying to plot graphs in epoch {epoch}: {e}\n\tThe lcag matrices were:\n\t{labels.numpy()}\n\tand\n\t{logits.cpu().detach().numpy()}"
                            )

                        model_state_dict = self.model.state_dict()
                        optimizer_state_dict = self.optimizer.state_dict()
                        checkpoint = {
                            "start_epoch": epoch,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer_state_dict,
                        }

                    if self.log_gradients and mode == "train":
                        all_grads.append(scale * epoch_grad)

                    mlflow.log_metric(
                        key=f"{mode}_accuracy", value=epoch_acc, step=epoch
                    )
                    mlflow.log_metric(key=f"{mode}_loss", value=epoch_loss, step=epoch)

                    # learning rate scheduling
                    self.scheduler.step()

        except GradientsNanException as e:
            log.error(f"Gradients became NAN during training\n{e}")
        except Exception as e:
            log.error(f"Exception occured during training\n{e}\n")
            traceback.print_exc()

        # quickly print the gradients..
        if self.log_gradients and len(all_grads) > 0:
            g_plt = self.plotGradients(all_grads, figsize=(16, 12))
            mlflow.log_figure(g_plt.gcf(), f"gradients.png")
            
        # In case we didn't even calculated a single sample
        if result == None:
            result = self.model

        log.info("Saving model and optimizer data")
        if checkpoint is None:
            try:
                model_state_dict = self.model.state_dict()
            except:
                model_state_dict = {}
            try:
                optimizer_state_dict = self.optimizer.state_dict()
            except:
                optimizer_state_dict = {}

            checkpoint = {
                "start_epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
            }

        mlflow.log_dict(model_state_dict, f"model.yml")
        mlflow.log_dict(optimizer_state_dict, f"optimizer.yml")

        return {
            "trained_model": result,
            "checkpoint": checkpoint,
            "gradients": all_grads,
        }

    def plotGradients(self, epoch_gradients, figsize=(16, 12)):

        X = [i for i in range(len(epoch_gradients[0]))]
        Y = [i for i in range(len(epoch_gradients))]
        Z = t.stack(epoch_gradients)

        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = heatmap(
            Z,
            Y,
            X,
            ax=ax,
            cmap="magma_r",
            cbarlabel=f"Gradients Normalized",
            axis_labels=("Parameters", "Epochs"),
            title="Gradient Values over Epochs",
        )
        # texts = annotate_heatmap(im, valfmt="{x:.1f} s")
        fig.tight_layout()
        return plt

    def plotBatchGraphs(self, batch_logits, batch_ref, rows=4, cols=2):
        fig, ax = plt.subplots(
            rows, cols, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1]}
        )
        fig.tight_layout()
        it = 0
        for logits, lcag_ref in zip(batch_logits, batch_ref):
            lcag = logits.max(0)[1]
            graph = GraphVisualization()

            graph.lca2graph(lcag)
            plt.sca(ax[it][0])
            if len(graph.visual) > 0:  # can happen if invalid graph -> no edge added
                graph.visualize(opt="max", ax=ax[it][0])

            graph_ref = GraphVisualization()

            graph_ref.lca2graph(lcag_ref)
            plt.sca(ax[it][1])
            graph_ref.visualize(opt="max", ax=ax[it][1])

            if it * cols > rows:
                break

            it += 1

        return plt

    def edge_accuracy(self, logits: t.Tensor, labels: t.Tensor) -> float:
        # logits: [Batch, Classes, LCA_0, LCA_1]
        probs = logits.softmax(1)  # get softmax for probabilities
        preds = probs.max(1)[1]  # find maximum across the classes (batches are on 0)

        correct = 0.0
        for batch_label, batch_preds in zip(labels, preds):
            correct += (batch_label == batch_preds).float().sum()/(labels.size(1) * labels.size(2)) # divide by the size of the matrix

        return correct / labels.size(0) # divide by the batch size
