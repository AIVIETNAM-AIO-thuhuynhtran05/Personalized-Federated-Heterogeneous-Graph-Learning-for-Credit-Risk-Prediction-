"""Relation-specific mean message passing with a customer classification head."""
import torch
from torch import nn


class HeteroGNN(nn.Module):
    def __init__(self, dimensions, relations, hidden=32, layers=2):
        super().__init__()
        self.relations = tuple(relations)
        self.input = nn.ModuleDict({node: nn.Linear(dim, hidden) for node, dim in dimensions.items()})
        self.self_layers = nn.ModuleList([
            nn.ModuleDict({node: nn.Linear(hidden, hidden) for node in dimensions}) for _ in range(layers)])
        self.messages = nn.ModuleList([
            nn.ModuleDict({relation: nn.Linear(hidden, hidden, bias=False) for relation in self.relations})
            for _ in range(layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, features, edges):
        hidden = {node: torch.relu(self.input[node](value)) for node, value in features.items()}
        for self_layer, messages in zip(self.self_layers, self.messages):
            updated = {node: self_layer[node](value) for node, value in hidden.items()}
            for relation in self.relations:
                source, _, target = relation.split("__")
                edge = edges[relation]
                if edge.shape[1] == 0:
                    continue
                message = messages[relation](hidden[source][edge[0]])
                aggregate = torch.zeros_like(updated[target]).index_add_(0, edge[1], message)
                degree = torch.bincount(edge[1], minlength=len(aggregate)).clamp_min(1).unsqueeze(1)
                updated[target] = updated[target] + aggregate / degree
            hidden = {node: torch.relu(value) for node, value in updated.items()}
        return self.head(hidden["customer"]).squeeze(1)
