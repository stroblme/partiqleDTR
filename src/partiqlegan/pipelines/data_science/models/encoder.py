from .base import MLP
from torch import nn
import torch
from torch import Tensor
from .gnn import GNN


class GNNENC(GNN):
    """
    Encoder of NRI. A combination of MLPEncoder and CNNEncoder from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 do_prob: float=0., factor: bool=False,#was true in original papaer
                 reducer: str='mlp'):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension, i.e., number of edge types
            do_prob: rate of dropout, default: 0
            factor: using a factor graph or not, default: True
            reducer: using an MLP or an CNN to reduce edge representations over multiple steps
        """
        super(GNNENC, self).__init__()
        self.factor = factor
        assert reducer.lower() in {'mlp', 'cnn'}
        self.reducer = reducer.lower()

        self.emb = MLP(n_in=n_in, n_hid=n_hid, n_out=2*n_hid, do_prob=do_prob)
        
        self.e2n = MLP(n_hid, n_hid, n_hid, do_prob)

        self.n2e_i = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.n2e_o = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def reduce_mlp(self, x: Tensor, es: Tensor):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        # x = x.view(x.size(0), x.size(1), -1)
        x_emb = self.emb(x)

        n_cat = x_emb.shape[0]
        x_0 = x_emb
        for i in range(n_cat-1):
            x_emb = torch.cat([x_emb, x_0], dim=0)
        z = x_emb
        # z = self.message(x_emb, es)
        return z


    def forward(self, x: Tensor, es: Tensor) -> Tensor:
        """
        Given the historical node states, output the K-dimension edge representations ready for relation prediction.

        Args:
            x: [batch, step, node, dim], node representations
            es: [2, E], edge list

        Return:
            z: [E, batch, dim], edge representations
        """
        x = x.permute(1, 0, -1).contiguous()
        # x: [batch, step, node, dim] -> [node, batch, step, dim]

        z = self.reduce_mlp(x, es)
        # z = self.emb(x.view(x.size(0), x.size(1), -1))

        z = self.n2e_i(z)
        z_skip = z
        # if self.factor:
        #     h = self.aggregate(z, col, size)
        #     h = self.e2n(h)
        #     z, _, __ = self.message(h, es)
        #     # skip connection
        #     z = torch.cat((z, z_skip), dim=2)
        #     z = self.n2e_o(z)
        # else:
        z = self.e2n(z)
        z = torch.cat((z, z_skip), dim=2)
        z = self.n2e_o(z)


        z = self.fc_out(z)
        return z
