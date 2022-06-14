from .base import MLP
from torch import nn
import torch
from torch import Tensor
import math
from .gnn import GNN
class CNN(nn.Module):
    """
    CNN from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float=0.):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension
        """
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_in, n_hid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(do_prob),
            # nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                        #  dilation=1, return_indices=False,
                        #  ceil_mode=False),
            # nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm1d(n_hid)
        )
        self.out = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.att = nn.Conv1d(n_hid, 1, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Args:
            inputs: [batch * E, dim, step], raw edge representations at each step

        Return:
            edge_prob: [batch * E, dim], edge representations over all steps with the step-dimension reduced
        """
        x = self.cnn(inputs)
        pred = self.out(x)
        attention = self.att(x).softmax(2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob

class GNNENC(GNN):
    """
    Encoder of NRI. A combination of MLPEncoder and CNNEncoder from https://github.com/ethanfetaya/NRI/modules.py.
    """
    def __init__(self, n_in: int, n_hid: int, n_out: int,
                 dropout_rate: float=0., factor: bool=False#was true in original papaer
                ):
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


        #-- CNN
        self.cnn = CNN(2*n_in, n_hid, n_hid*2, dropout_rate)
        


        #n_in=4, n_hid=1024

        #-- MLP
        # self.emb = MLP(n_in=n_in, n_hid=n_hid, n_out=n_hid, do_prob=dropout_rate)
        
        # self.e2n = MLP(n_hid, n_hid, n_hid, dropout_rate)

        self.n2e_i = MLP(2*n_hid, n_hid, n_hid, dropout_rate)

        self.interm_mlp_a = [MLP(n_hid, n_hid, n_hid, dropout_rate) for i in range(3)] #cnn
        self.e2n_s = [MLP(2*n_hid, n_hid, n_hid, dropout_rate) for i in range(3)]

        self.interm_mlp_b = [MLP(n_hid, n_hid, n_hid, dropout_rate) for i in range(3)] #cnn
        self.n2e_s = [MLP(n_hid * 2, n_hid, n_hid, dropout_rate) for i in range(3)]




        
        self.fc_out = nn.Linear(n_hid * 2, n_out)
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
        # infer the step and dim size (which is equal to just the dim in our case)
        x = x.view(x.size(0), x.size(1), -1)

        # apply the embedding (MLP, transforming n_in into n_hid)
        x = self.emb(x)
        z, col, size = self.message(x, es)
        return z, col, size

    def reduce_cnn(self, x, es):
        """
        Args:
            x: [node, batch, step, dim]
            es: [2, E]
        
        Return:
            z: [E, batch, dim]
            col: [E]
            size: int
        """
        # z: [E, batch, step, dim * 2]
        # x = x.view(x.size(0), x.size(2), x.size(1))
        z, col, size = self.message(x, es, option="i2o")
        z = z.transpose(3, 2).contiguous()
        # z: [E * batch, dim * 2, step]
        z = z.view(-1, z.size(2), z.size(3))
        # z = x.view(x.size(0), x.size(1), -1)

        z = self.cnn(z)
        z = z.view(len(col), x.size(1), -1)
        return z, col, size

    def forward(self, x: Tensor, es: Tensor) -> Tensor:
        """
        Given the historical node states, output the K-dimensionz.shape edge representations ready for relation prediction.

        Args:
            x: [batch, step, node, dim], node representations
            es: [2, E], edge list

        Return:
            z: [E, batch, dim], edge representations
        """
        # x = x.permute(1, 0, -1).contiguous()
        x = x.view(x.size(1), x.size(0), 1, x.size(-1))
        # x: [batch, step, node, dim] -> [node, batch, step, dim]

        # z = self.reduce_mlp(x, es)[0]
        z = self.reduce_cnn(x, es)[0]


        # z = self.emb(x.view(x.size(0), x.size(1), -1))

        z = self.n2e_i(z)
        z_skip_g = z

            # z = self.n2e_i_s[i](z)
        for i in range(3):
            z_skip = z
            z = self.interm_mlp_a[i](z)
            z = torch.cat((z, z_skip), dim=2)
            z = self.e2n_s[i](z)
            z_skip = z
            z = self.interm_mlp_b[i](z)
            z = torch.cat((z, z_skip), dim=2)
            z = self.n2e_s[i](z)

        z = torch.cat((z, z_skip_g), dim=2)



        z = self.fc_out(z)
        return z
