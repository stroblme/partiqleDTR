import time

import matplotlib.pyplot as plt

import torch as t
from torch.nn.parallel import DataParallel
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, data, normalize=False):
        dmax = 0
        dmin = 1
        self.data = data
        if normalize:
            # for i, event in enumerate(data.x):
            #     dmax = event.max() if event.max() > dmax else dmax
            #     dmin = event.min() if event.min() < dmin else dmin
            # for i, event in enumerate(data.x):
            #     self.data.x[i] = (event-dmin)/(dmax-dmin)
            # dmax = 0
            # dmin = 1
            # for i, event in enumerate(data.x):
            #     dmax = event.max() if event.max() > dmax else dmax
            #     dmin = event.min() if event.min() < dmin else dmin
            # assert dmax==1 and dmin==0

            for i, event in enumerate(data.x):
                dmax = event.max()
                dmin = event.min()

                self.data.x[i] = (event - dmin) / (dmax - dmin)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


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
        plot_mode: str = "val",
        detectAnomaly: bool = False,
        device: str = "cpu",
        n_fsps=-1,
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
            p.register_hook(lambda grad: t.clamp(grad, -1000, 1000))
            p.register_hook(lambda grad: t.where(grad < 1e-10, t.rand(1) * 1e-9, grad))

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
        self.detectAnomaly = detectAnomaly  # TODO: introduce as parameter if helpful

    def train(self):
        if self.detectAnomaly:
            log.info(f"Anomaly detection enabled")
            t.autograd.set_detect_anomaly(True)

        log.info(f"Training started with a batch size of {self.batch_size}")
        result = None
        best_acc = 0
        all_grads = []

        try:  # catch things like gradient nan exceptions
            for epoch in range(1, 1 + self.epochs):
                for mode in ["train", "val"]:
                    data_batch = DataLoader(
                        DataWrapper(self.data[mode], normalize=self.normalize),
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=rel_pad_collate_fn,
                    )  # to handle varying input size

                    epoch_loss = 0.0
                    epoch_acc = 0.0
                    epoch_grad = t.zeros(len([p for p in self.model.parameters()][0]))

                    log.info(
                        f"Running epoch {epoch} in mode {mode} over {len(data_batch)} samples"
                    )
                    for i, (states, labels) in enumerate(data_batch):
                        sample_start = time.time()
                        states = [s.to(self.device) for s in states]
                        labels = labels.to(self.device)
                        scale = 1 / labels.size(
                            1
                        )  # get the scaling dependend on the number of classes

                        if mode == "train":
                            self.model.train()  # set the module in training mode

                            logits = self.model(states)
                            loss = cross_entropy(logits, labels, ignore_index=-1)
                            acc = self.edge_accuracy(logits, labels)

                            # do the actual optimization
                            self.opt.zero_grad()
                            loss.backward()
                            self.opt.step()

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

                                loss = cross_entropy(logits, labels, ignore_index=-1)
                                acc = self.edge_accuracy(logits, labels)
                        elif mode == "test":
                            self.model.eval()  # trigger evaluation forward mode
                            with t.no_grad():  # disable autograd in tensors

                                logits = self.model(states)

                                loss = cross_entropy(logits, labels, ignore_index=-1)
                                acc = self.edge_accuracy(logits, labels)
                        else:
                            log.error("Unknown mode")

                        epoch_loss += scale * loss.item()
                        epoch_acc += scale * acc.item()

                        if acc > best_acc and mode == self.plot_mode:
                            # update the current best model when approaching a higher accuray
                            best_acc = acc
                            result = self.model
                            try:
                                c_plt = self.plotBatchGraphs(logits.cpu(), labels)
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
                                "epoch": epoch,
                                "loss": loss,
                                "model_state_dict": model_state_dict,
                                "optimizer_state_dict": optimizer_state_dict
                            }



                        if mode == "train":
                            log.debug(
                                f"Sample evaluation in epoch {epoch}, iteration {i} took {time.time() - sample_start} seconds. Loss was {scale*loss.item()}"
                            )

                    epoch_loss /= len(
                        data_batch
                    )  # to the already scaled loss, apply the batch size scaling
                    epoch_acc /= len(
                        data_batch
                    )  # to the already scaled accuracy, apply the batch size scaling

                    if mode == "train":
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
            log.error(f"Exception occured during training\n{e}")

        # quickly print the gradients..
        if len(all_grads) > 0:
            g_plt = self.plotGradients(all_grads, figsize=(16, 12))
            mlflow.log_figure(g_plt.gcf(), f"gradients.png")

        if result == None:
            result = self.model

        return {
            "trained_model":result, 
            "checkpoint":checkpoint,
            "gradients":all_grads
        }

    def plotGradients(self, epoch_gradients, figsize=(16, 12)):

        X = [i for i in range(len(epoch_gradients))]
        Y = [i for i in range(len(epoch_gradients[0]))]
        Z = t.stack(epoch_gradients)

        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = heatmap(
            Z,
            X,
            Y,
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
        preds = probs.max(1)[1]  # find maximum across the classes
        correct = (labels == preds).sum().float()
        return correct / (labels.size(1) * labels.size(2))
