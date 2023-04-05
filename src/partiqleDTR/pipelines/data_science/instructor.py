import copy
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
from .gradients_visualization import heatmap, heatmap_3d, scatter_line
from .circuit_gradient_visualization import draw_gradient_circuit

from typing import Dict

import logging

log = logging.getLogger(__name__)


class GradientsNanException(RuntimeError):
    pass


class DataWrapper(Dataset):
    """
    A wrapper for t.utils.data.Dataset.
    """

    def __init__(
        self, data, normalize="", normalize_individually=True, zero_mean=False
    ):

        # normalize_individually = True
        # zero_mean = False

        self.data = data
        if normalize == "one":
            dmax = 0
            dmin = 0
            if not normalize_individually:
                for i, event in enumerate(data.x):
                    dmax = event.max() if event.max() > dmax else dmax
                    dmin = event.min() if event.min() < dmin else dmin

            for i, event in enumerate(data.x):
                if normalize_individually:
                    dmax = event.max()
                    dmin = event.min()

                if zero_mean:
                    self.data.x[i] = (event - (dmax - dmin) / 2) / (dmax - dmin)
                else:
                    # self.data.x[i] = (event - dmin) / (dmax - dmin)
                    self.data.x[i] = (event) / (dmax - dmin)
        elif normalize == "smartone":
            dmax_p = 0
            dmin_p = 0
            dmax_e = 0
            dmin_e = 0
            if not normalize_individually:
                for i, event in enumerate(data.x):
                    dmax_p = (
                        event[:, :3].max() if event[:, :3].max() > dmax_p else dmax_p
                    )
                    dmin_p = (
                        event[:, :3].min() if event[:, :3].min() < dmin_p else dmin_p
                    )
                    dmax_e = event[:, 3].max() if event[:, 3].max() > dmax_e else dmax_e
                    dmin_e = event[:, 3].min() if event[:, 3].min() < dmin_e else dmin_e

            for i, event in enumerate(data.x):
                if normalize_individually:
                    dmax_p = event[:, :3].max()
                    dmin_p = event[:, :3].min()
                    dmax_e = event[:, 3].max()
                    dmin_e = event[:, 3].min()

                if zero_mean:
                    self.data.x[i][:, :3] = (event[:, :3] - (dmax_p - dmin_p) / 2) / (
                        dmax_p - dmin_p
                    )
                    # self.data.x[i][3] = (event[3] - (dmax_e - dmin_e)/2) / (dmax_e - dmin_e)
                    self.data.x[i][:, 3] = (event[:, 3]) / (
                        dmax_e - dmin_e
                    )  # it does not make sense to shift the energy as this could result in negative energy values
                else:
                    self.data.x[i][:, :3] = (event[:, :3]) / (dmax_p - dmin_p)
                    self.data.x[i][:, 3] = (event[:, 3]) / (dmax_e - dmin_e)

        elif normalize == "zmuv":
            for i, event in enumerate(data.x):
                # adj_mean = t.repeat_interleave(t.Tensor([data.mean]), event.shape[0], dim=0)
                # adj_std = t.repeat_interleave(t.Tensor([data.std]), event.shape[0], dim=0)
                adj_mean = np.repeat([data.mean], event.shape[0], axis=0)
                adj_std = np.repeat([data.std], event.shape[0], axis=0)
                self.data.x[i] = (event - adj_mean) / adj_std

        # fill diagonal with zeros to ignore in loss
        for i, lcag in enumerate(data.y):
            np.fill_diagonal(self.data.y[i], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def calculate_class_weights(
    dataloader, num_classes, num_batches=100, amp_enabled=False
):
    """Calculates class weights based on num_batches of the dataloader

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
    weights = t.nan_to_num(weights, posinf=float("nan"))
    # And normalise to sum to 1
    weights = weights / weights.nansum()
    # Finally, assign default value to any that were missing during calculation time
    weights = t.nan_to_num(weights, nan=1)

    return weights

class SplitOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        combined_state_dict = {}
        for op in self.optimizers:
            combined_state_dict |= op.state_dict()

        return combined_state_dict

    def load_state_dict(self, combined_state_dict):
        raise NotImplementedError("Loading state dict into multiple optimizers not supported yet.")


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
        normalize: str,
        normalize_individually: bool,
        zero_mean: bool,
        plot_mode: str = "val",
        plotting_rows: int = 4,
        log_gradients: bool = False,
        detectAnomaly: bool = False,
        device: str = "cpu",
        n_fsps=-1,
        n_classes=-1,
        gradients_clamp=1000,
        gradients_spreader=1e-10,
        torch_seed=1111,
        gradient_curvature_threshold=1e-10,
        gradient_curvature_history=2,
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

        t.manual_seed(torch_seed)

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
                    grad < float(gradients_spreader),
                    t.rand(1) * float(gradients_spreader) * 1e1,
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
        self.optimizer = SplitOptimizer(
            t.optim.Adam(
                self.model.quantum_layer.parameters(), lr=learning_rate, amsgrad=False
            ),
            t.optim.Adam(
                self.model.gnn.parameters(), lr=learning_rate, amsgrad=False
            )
        )

        self.gradient_curvature_threshold = float(gradient_curvature_threshold)
        self.gradient_curvature_history = int(gradient_curvature_history)

        self.n_classes = n_classes

        # learning rate scheduler, same as in NRI

        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        self.scheduler = StepLR(
            self.optimizer.optimizers[1], step_size=learning_rate_decay, gamma=gamma
        ) # use secheduling only for the classical optim
        self.detectAnomaly = detectAnomaly  # TODO: introduce as parameter if helpful

    def train(self, start_epoch=1, enabled_modes=["train", "val"]):
        t.autograd.set_detect_anomaly(self.detectAnomaly)

        log.info(
            f"Starting loops from epoch {start_epoch} using modes {enabled_modes}."
        )
        result = None
        best_acc = 0
        checkpoint = None

        if enabled_modes == ["val"]:
            self.epochs = 1

        # TODO: logging gradients currently only enabled for qgnn
        self.log_gradients = self.log_gradients and self.model._get_name() == "qgnn"

        if self.log_gradients:
            all_grads = []

        error_raised = True

        try:  # catch things like gradient nan exceptions
            for epoch in range(start_epoch, 1 + self.epochs):
                logits_for_plotting = []
                labels_for_plotting = []
                for mode in enabled_modes:
                    data_batch = DataLoader(
                        DataWrapper(
                            self.data[mode],
                            normalize=self.normalize,
                            normalize_individually=self.normalize_individually,
                            zero_mean=self.zero_mean,
                        ),
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=rel_pad_collate_fn,
                    )  # to handle varying input size

                    # might seem better to be put in data processing or sth.
                    weights = calculate_class_weights(
                        data_batch, self.n_classes, len(data_batch), False
                    )
                    weights = weights.to(self.device)

                    epoch_loss = 0.0
                    epoch_acc = 0.0
                    epoch_logic_acc = 0.0
                    epoch_perfect_lcag = 0.0

                    if self.log_gradients:
                        epoch_grad = []

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
                        # scale = 1 / labels.size(
                        #     1
                        # )  # get the scaling dependent on the number of classes

                        if mode == "train":
                            self.model.train()  # set the module in training mode

                            logits = self.model(states)
                            loss = cross_entropy(
                                logits, labels, weight=weights, ignore_index=-1
                            )
                            acc = self.edge_accuracy(logits, labels, ignore_index=-1)
                            logic_acc = self.logic_accuracy(
                                logits, labels, ignore_index=-1
                            )
                            perfect_lcag = self.perfect_lcag(
                                logits, labels, ignore_index=-1
                            )

                            # self.plotBatchGraphs(logits, labels)

                            # do the actual optimization
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            if self.log_gradients:
                                # raise Exception("test")
                                epoch_grad.append(
                                    self.model.quantum_layer.weight.grad
                                )  # n_batch_samples, n_weights

                                if t.any(
                                    t.isnan(self.model.quantum_layer.weight.grad)
                                ):  # TODO: we are checking for "all" instead of "any" since there were cases where the gradients became nan but training successfully continued (investigate in samples!)
                                    log.error(
                                        f"At least one gradient became nan in epoch {epoch} after iteration {i}.\nInput was\n{states}.\nPredicted was\n{logits}.\nGradients are\n{self.model.quantum_layer.weight.grad}"
                                    )
                                else:
                                    log.debug(
                                        f"Gradients in epoch {epoch}, iteration {i}: {self.model.quantum_layer.weight.grad}"
                                    )

                            labels = labels.cpu()
                            if labels.numpy().min() < -1:
                                raise Exception(
                                    f"Found graph with negative values: {labels.numpy()}"
                                )
                        elif mode == "val":
                            self.model.eval()  # trigger evaluation forward mode
                            with t.no_grad():  # disable autograd in tensors

                                logits = self.model(states)

                                loss = cross_entropy(
                                    logits, labels, weight=weights, ignore_index=-1
                                )
                                acc = self.edge_accuracy(
                                    logits, labels, ignore_index=-1
                                )
                                logic_acc = self.logic_accuracy(
                                    logits, labels, ignore_index=-1
                                )
                                perfect_lcag = self.perfect_lcag(
                                    logits, labels, ignore_index=-1
                                )
                        elif mode == "test":
                            self.model.eval()  # trigger evaluation forward mode
                            with t.no_grad():  # disable autograd in tensors

                                logits = self.model(states)

                                loss = cross_entropy(
                                    logits, labels, weight=weights, ignore_index=-1
                                )
                                acc = self.edge_accuracy(
                                    logits, labels, ignore_index=-1
                                )
                                logic_acc = self.logic_accuracy(
                                    logits, labels, ignore_index=-1
                                )
                                perfect_lcag = self.perfect_lcag(
                                    logits, labels, ignore_index=-1
                                )
                        else:
                            log.error("Unknown mode")

                        epoch_loss += (
                            1 / self.n_classes
                        ) * loss.item()  # access via .item() to get a float value instead of a tensor obj
                        epoch_acc += (
                            acc.item()
                        )  # access via .item() to get a float value instead of a tensor obj
                        epoch_logic_acc += (
                            logic_acc.item()
                        )  # access via .item() to get a float value instead of a tensor obj
                        epoch_perfect_lcag += perfect_lcag  # don't scale accuracy and perfect_lcag as they are not class dependent

                        if mode == "train":
                            log.debug(
                                f"Sample evaluation in epoch {epoch}, iteration {i} took {time.time() - sample_start} seconds. Loss was {(1 / self.n_classes)*loss.item()}"
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
                    epoch_logic_acc /= len(
                        data_batch
                    )  # to the already scaled accuracy, apply the number of all iterations (no. of mini batches)
                    epoch_perfect_lcag /= len(
                        data_batch
                    )  # to the already scaled perfect_lcag, apply the number of all iterations (no. of mini batches)

                    if epoch_acc > best_acc and mode == self.plot_mode:
                        # update the current best model when approaching a higher accuray
                        best_acc = epoch_acc
                        result = self.model
                        try:
                            assert len(logits_for_plotting) == len(labels_for_plotting)
                            plotting_indices = [
                                random.choice(range(len(logits_for_plotting)))
                                for i in range(self.plotting_rows)
                            ]

                            selected_logits = [
                                logits_for_plotting[i].cpu().detach()
                                for i in plotting_indices
                            ]
                            selected_labels = [
                                labels_for_plotting[i] for i in plotting_indices
                            ]
                            logits_string = "\n\n\n".join(
                                "\n".join(
                                    "\t".join(f"{x}" for x in y) for y in a_b.max(0)[1]
                                )
                                for a_b in selected_logits
                            )
                            labels_string = "\n\n\n".join(
                                "\n".join("\t".join(f"{x}" for x in y) for y in a_b)
                                for a_b in selected_labels
                            )

                            c_plt = self.plotBatchGraphs(
                                selected_logits,
                                selected_labels,
                                rows=self.plotting_rows,
                                cols=2,
                            )
                            mlflow.log_figure(
                                c_plt.gcf(), f"{mode}_e{epoch}_sample_graph.png"
                            )
                            mlflow.log_text(
                                logits_string, f"{mode}_e{epoch}_logits.txt"
                            )
                            mlflow.log_text(
                                labels_string, f"{mode}_e{epoch}_labels.txt"
                            )

                        except Exception as e:
                            log.error(
                                f"Exception occured when trying to plot graphs in epoch {epoch}: {e}\n\tThe lcag matrices were:\n\n{labels_string}\n\tand\n\n{logits_string}\n{traceback.print_exc()}"
                            )

                        model_state_dict = self.model.state_dict()
                        optimizer_state_dict = self.optimizer.state_dict()
                        checkpoint = {
                            "start_epoch": epoch,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer_state_dict,
                        }

                    if self.log_gradients and mode == "train":
                        all_grads.append(
                            t.stack(epoch_grad)
                        )  # n_epochs, n_batch_samples, n_weights

                        if self.gradient_curvature_history > 0:
                            selected_parameters = self.parameter_pruning(
                                self.model.var_params, t.stack(all_grads).mean(dim=1)
                            )  # use mean over batch samples

                            mlflow.log_metric(key=f"num_selected_params", value=len(selected_parameters), step=epoch)

                            self.model.quantum_layer.neural_network.set_selected_parameters(
                                selected_parameters
                            )

                    mlflow.log_metric(key=f"{mode}_loss", value=epoch_loss, step=epoch)
                    mlflow.log_metric(
                        key=f"{mode}_accuracy", value=epoch_acc, step=epoch
                    )
                    mlflow.log_metric(
                        key=f"{mode}_logic_accuracy", value=epoch_logic_acc, step=epoch
                    )
                    mlflow.log_metric(
                        key=f"{mode}_perfect_lcag", value=epoch_perfect_lcag, step=epoch
                    )
                    # learning rate scheduling
                    self.scheduler.step()

            error_raised = False

        except GradientsNanException as e:
            log.error(f"Gradients became NAN during training\n{traceback.print_exc()}")
        except Exception as e:
            log.error(f"Exception occured during training\n{traceback.print_exc()}\n")

        # quickly print the gradients..
        if self.log_gradients and len(all_grads) > 0:
            all_grads = t.stack(all_grads)

            g_plt, g3d_plt = self.plotGradients(
                all_grads.mean(dim=1)
            )  # use mean over batch samples
            mlflow.log_figure(g_plt, f"gradients.html")
            mlflow.log_figure(g3d_plt, f"gradients_3d.html")

            gc_plt = self.gradient_pqc_viz(
                self.model, all_grads.mean(dim=1)
            )  # use mean over batch samples
            mlflow.log_figure(
                gc_plt,
                "circuit_gradients.png",
            )

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

        if error_raised:
            raise RuntimeError(
                "Training did not complete successfully. Model and optimizer state dictionaries were saved though."
            )

        return {
            "trained_model": result,
            "checkpoint": checkpoint,
            "gradients": all_grads.numpy(),
        }

    def parameter_pruning(self, parameters, gradients: t.Tensor):
        # gradients should be of shape [epochs, n_weights]

        curvature = gradients.diff(dim=0)

        sel_params = []

        if (
            curvature.size(0) == 0
            or curvature.size(0) < self.gradient_curvature_history
        ):
            sel_params = parameters
        else:
            # TODO: verify that this still works if history is 1
            for param, curv in zip(
                parameters, curvature[-self.gradient_curvature_history :].mean(dim=0)
            ):  # we only use the diff between the current and previous epoch
                # if curvature is greater than a threshold, the parameter seems to be updated frequently -> don't prune it
                if curv.mean().item() > self.gradient_curvature_threshold:
                    sel_params.append(param)

        return sel_params

    def plotGradients(self, epoch_gradients: t.Tensor):
        # gradients should be of shape [epochs, n_weights]

        X = [i for i in range(len(epoch_gradients[0]))]
        Y = [i for i in range(len(epoch_gradients))]
        Z = epoch_gradients.abs()  # use absolute value of gradients

        fig = heatmap(
            Z,
            Y,
            X,
            cmap="magma_r",
            cbarlabel=f"Gradients (Data Mean, Absolute Values)",
            axis_labels=("Parameters", "Epochs"),
            title="Gradient Values over Epochs",
        )

        fig3d = heatmap_3d(
            Z,
            Y,
            X,
            cmap="magma_r",
            cbarlabel=f"Gradients (Data Mean, Absolute Values)",
            axis_labels=("Parameters", "Epochs"),
            title="Gradient Values over Epochs",
        )

        return fig, fig3d

    def plotSelectedParams(self, num_selected_params):
        X = [i for i in range(len(num_selected_params))]
        Y = num_selected_params

        fig = scatter_line(Y, X, "Selected Parameters")

        return fig

    # def plotGradients(self, epoch_gradients:t.Tensor, figsize=(16, 12)):
    #     # gradients should be of shape [epochs, n_weights]

    #     X = [i for i in range(len(epoch_gradients[0]))]
    #     Y = [i for i in range(len(epoch_gradients))]
    #     Z = epoch_gradients.abs()   # use absolute value of gradients

    #     fig, ax = plt.subplots(figsize=figsize)
    #     im, cbar = heatmap(
    #         Z,
    #         Y,
    #         X,
    #         ax=ax,
    #         cmap="magma_r",
    #         cbarlabel=f"Gradients (Data Mean, Absolute Values)",
    #         axis_labels=("Parameters", "Epochs"),
    #         title="Gradient Values over Epochs",
    #     )
    #     # texts = annotate_heatmap(im, valfmt="{x:.1f} s")
    #     fig.tight_layout()
    #     return plt

    def gradient_pqc_viz(self, model, gradients: t.Tensor):
        # gradients should be of shape [epochs, n_weights]

        # gradients = t.stack(gradients)
        circuit = model.qc

        plotly_figure = draw_gradient_circuit(gradients=gradients, circuit=circuit)

        return plotly_figure

    def plotBatchGraphs(self, batch_logits, batch_ref, rows=4, cols=2):
        fig, ax = plt.subplots(
            rows, cols, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1]}
        )
        fig.tight_layout()

        for it, (logits, lcag_ref) in enumerate(zip(batch_logits, batch_ref)):
            probs = logits.softmax(0)  # TODO: verify this is correct
            lcag = probs.max(0)[1]
            assert lcag_ref.shape == lcag.shape

            lcag = t.where(lcag_ref == -1, lcag_ref, lcag)

            graph = GraphVisualization()

            graph.lca2graph(lcag)
            plt.sca(ax[it][0])
            if len(graph.visual) > 0:  # can happen if invalid graph -> no edge added
                graph.visualize(opt="max", ax=ax[it][0])

            graph_ref = GraphVisualization()

            graph_ref.lca2graph(lcag_ref)
            plt.sca(ax[it][1])
            graph_ref.visualize(opt="max", ax=ax[it][1])

            # if it >= rows:
            #     break

        return plt

    def logic_accuracy(
        self, logits: t.Tensor, labels: t.Tensor, ignore_index: int = None
    ) -> float:
        def two_child_fix(lcag):
            max_c = lcag.max().int()

            def convToPair(pair: t.Tensor):
                return (int(pair[0]), int(pair[1]))

            working_lcag = copy.deepcopy(lcag)
            while working_lcag.max() > 0:
                next_c = 0
                for c in range(1, max_c + 1):
                    directPairs = list((working_lcag == c).nonzero())
                    directPairs.sort(key=lambda dp: sum(dp))

                    if len(directPairs) > 0:
                        next_c = c
                        break

                if next_c == 1:
                    working_lcag -= 1
                    continue

                for pair in directPairs:
                    index = convToPair(pair)
                    lcag[index] -= 1

                working_lcag = copy.deepcopy(lcag)

            return lcag

        correct = 0.0
        for label, logit in zip(labels, logits):
            # logits: [Batch, Classes, LCA_0, LCA_1]
            probs = logit.softmax(0)  # get softmax for probabilities
            prediction = probs.max(0)[
                1
            ]  # find maximum across the classes (batches are on 0)

            if ignore_index is not None:
                # set everything to -1 which is not relevant for grading
                prediction = t.where(label == ignore_index, label, prediction)

            # test_lcag_a = t.Tensor([    [-1,  1,  2,  2],
            #                             [ 1, -1,  2,  1],
            #                             [ 2,  2, -1,  2],
            #                             [ 2,  1,  2, -1]])
            # test_lcag_b = t.Tensor([    [-1,  1,  2,  2],
            #                             [ 1, -1,  2,  0],
            #                             [ 2,  2, -1,  2],
            #                             [ 2,  0,  2, -1]])
            # test_lcag_c = t.Tensor([    [-1,  1,  3,  3],
            #                             [ 1, -1,  3,  3],
            #                             [ 3,  3, -1,  1],
            #                             [ 3,  3,  1, -1]])
            # test_lcag_d = t.Tensor([    [-1,  1,  3, -1],
            #                             [ 1, -1,  3, -1],
            #                             [ 3,  3, -1, -1],
            #                             [-1, -1, -1, -1]])

            # test_lcag_a = two_child_fix(test_lcag_a)
            # test_lcag_b = two_child_fix(test_lcag_b)
            # test_lcag_c = two_child_fix(test_lcag_c)
            # test_lcag_d = two_child_fix(test_lcag_d)
            prediction = two_child_fix(prediction)

            # which are the correct predictions
            a = label == prediction

            if ignore_index is not None:
                # create a mask hiding the irrelevant entries
                b = label != t.ones(label.shape) * ignore_index
            else:
                b = label == label  # simply create an "True"-matrix to hide the mask

            correct += (
                a == b
            ).float().sum() / b.sum()  # divide by the size of the matrix

        return correct / labels.size(0)  # divide by the batch size

    def edge_accuracy(
        self, logits: t.Tensor, labels: t.Tensor, ignore_index: int = None
    ) -> float:

        correct = 0.0
        for label, logit in zip(labels, logits):
            # logits: [Batch, Classes, LCA_0, LCA_1]
            probs = logit.softmax(0)  # get softmax for probabilities
            prediction = probs.max(0)[
                1
            ]  # find maximum across the classes (batches are on 0)
            if ignore_index is not None:
                # set everything to -1 which is not relevant for grading
                prediction = t.where(label == ignore_index, label, prediction)

            # which are the correct predictions
            a = label == prediction

            if ignore_index is not None:
                # create a mask hiding the irrelevant entries
                b = label != t.ones(label.shape) * ignore_index
            else:
                b = label == label  # simply create an "True"-matrix to hide the mask

            correct += (
                a == b
            ).float().sum() / b.sum()  # divide by the size of the matrix

        return correct / labels.size(0)  # divide by the batch size

    def perfect_lcag(
        self, logits: t.Tensor, labels: t.Tensor, ignore_index: int = None
    ) -> float:

        correct = 0.0
        for label, logit in zip(labels, logits):
            # logits: [Batch, Classes, LCA_0, LCA_1]
            probs = logit.softmax(0)  # get softmax for probabilities
            prediction = probs.max(0)[
                1
            ]  # find maximum across the classes (batches are on 0)

            if ignore_index is not None:
                # set everything to -1 which is not relevant for grading
                prediction = t.where(label == ignore_index, label, prediction)

            # which are the correct predictions
            a = label == prediction

            if ignore_index is not None:
                # create a mask hiding the irrelevant entries
                b = label != t.ones(label.shape) * ignore_index
            else:
                b = label == label  # simply create an "True"-matrix to hide the mask

            if (a == b).float().sum() / b.sum() == 1:
                correct += 1
            # log.info(f"Perfect lcag candidate was {(a == b).float().sum()/b.sum()}")

        return correct / labels.size(0)  # divide by the batch size
