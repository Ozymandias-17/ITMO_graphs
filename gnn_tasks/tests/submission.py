import torch
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops, scatter


class MyGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index):
        n = x.size(0)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=n)
        row, col = edge_index

        deg = scatter(x.new_ones(col.numel()), col, dim=0, dim_size=n, reduce="sum")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.lin(x)

        out = scatter(x[row] * norm.unsqueeze(-1), col, dim=0, dim_size=n, reduce="sum")

        return out + self.bias


class MySAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        neigh_mean = scatter(x[row], col, dim=0, dim_size=x.size(0), reduce="mean")

        return self.lin_neigh(neigh_mean) + self.lin_self(x)
