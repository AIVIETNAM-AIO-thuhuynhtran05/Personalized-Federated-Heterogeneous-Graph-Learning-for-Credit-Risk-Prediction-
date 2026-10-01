"""HeteroGNN cho bài toán phân loại node customer.

Mỗi node type có encoder riêng -> L tầng HeteroConv(SAGEConv theo từng edge type, có residual)
-> MLP trên embedding customer. Với L=2: tầng 1 đưa thông tin event -> prev và prev/bureau -> customer,
tầng 2 đưa prev (đã chứa thông tin event) -> customer. Tầng cuối chỉ tính các cạnh đi vào customer.
"""
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from src.graph.schema import CUSTOMER


class HeteroGNN(nn.Module):
    def __init__(self, in_dims: dict[str, int], edge_types: list[tuple], hidden: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.encoders = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU())
            for t, d in in_dims.items()})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            last = layer == num_layers - 1
            ets = [et for et in edge_types if not last or et[2] == CUSTOMER]
            self.convs.append(HeteroConv(
                {et: SAGEConv((hidden, hidden), hidden, aggr="mean") for et in ets}, aggr="sum"))
            self.norms.append(nn.ModuleDict({t: nn.LayerNorm(hidden) for t in {et[2] for et in ets}}))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict) -> torch.Tensor:
        h = {t: self.encoders[t](x) for t, x in x_dict.items()}
        for conv, norms in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict)
            h = {**h, **{t: h[t] + nn.functional.dropout(torch.relu(norms[t](o)), self.dropout,
                                                         self.training)
                         for t, o in out.items()}}
        return self.head(h[CUSTOMER]).squeeze(-1)
