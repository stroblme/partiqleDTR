from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob, batchnorm=True):
        super(MLP, self).__init__()

        self.batchnorm = batchnorm

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=True)
            # self.bn = nn.BatchNorm1d(n_out, momentum=0.1, track_running_stats=False)  # Use this to overfit
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm_layer(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        """
        Input: (b, l, c)
        Output: (b, l, d)
        """
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))  # (b, l, d)
        x = F.dropout(x, self.dropout_prob, training=self.training)  # (b, l, d)
        x = F.elu(self.fc2(x))  # (b, l, d)
        return self.batch_norm_layer(x) if self.batchnorm else x  # (b, l, d)


def generate_nri_blocks(
    dim_feedforward,
    batchnorm,
    dropout_rate,
    n_additional_mlp_layers,
    block_dim,
    n_blocks,
):
    # MLPs within NRI blocks
    # The blocks have minimum 1 MLP layer, and if specified they add more with a skip connection
    # List of blocks

    blocks = nn.ModuleList(
        [
            # List of MLP sequences within each block
            nn.ModuleList(
                [
                    # MLP layers before Edge2Node (start of block)
                    nn.ModuleList(
                        [
                            MLP(
                                dim_feedforward,
                                dim_feedforward,
                                dim_feedforward,
                                dropout_rate,
                                batchnorm,
                            ),
                            nn.Sequential(
                                *[
                                    MLP(
                                        dim_feedforward,
                                        dim_feedforward,
                                        dim_feedforward,
                                        dropout_rate,
                                        batchnorm,
                                    )
                                    for _ in range(n_additional_mlp_layers)
                                ]
                            ),
                            # This is what would be needed for a concat instead of addition of the skip connection
                            # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                        ]
                    ),
                    # MLP layers between Edge2Node and Node2Edge (middle of block)
                    nn.ModuleList(
                        [
                            MLP(
                                dim_feedforward,
                                dim_feedforward,
                                dim_feedforward,
                                dropout_rate,
                                batchnorm,
                            ),
                            nn.Sequential(
                                *[
                                    MLP(
                                        dim_feedforward,
                                        dim_feedforward,
                                        dim_feedforward,
                                        dropout_rate,
                                        batchnorm,
                                    )
                                    for _ in range(n_additional_mlp_layers)
                                ]
                            ),
                            # This is what would be needed for a concat instead of addition of the skip connection
                            # MLP(dim_feedforward * 2, dim_feedforward, dim_feedforward, dropout, batchnorm) if (block_additional_mlp_layers > 0) else None,
                        ]
                    ),
                    # MLP layer after Node2Edge (end of block)
                    # This is just to reduce feature dim after skip connection was concatenated
                    MLP(
                        block_dim,
                        dim_feedforward,
                        dim_feedforward,
                        dropout_rate,
                        batchnorm,
                    ),
                ]
            )
            for _ in range(n_blocks)
        ]
    )

    return blocks
