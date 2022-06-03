from torch import Tensor, nn
from ..utils.torch_extension import gumbel_softmax, sym_hard
# import config as cfg
import torch


class NRIModel(nn.Module):
    """Auto-encoder."""
    def __init__(self, encoder: nn.Module, es: Tensor, size: int):
        """
        Args:
            encoder: an encoder inferring relations
            decoder: an decoder predicting future states
            es: edge list
            size: number of nodes
        """
        super(NRIModel, self).__init__()
        self.enc = encoder
        self.es = es
        self.size = size

    def predict_relations(self, states: Tensor) -> Tensor:
        """
        Given historical node states, infer interacting relations.

        Args:
            states: [batch, step, node, dim]

        Return:
            prob: [E, batch, K]
        """
        # if not self.es.is_cuda:
            # self.es = self.es.cuda(states.device)
        logits = self.enc(states, self.es)
        prob = logits.softmax(-1)
        return prob

