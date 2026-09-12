# GNN mean theo từng loại quan hệ: chiếu đặc trưng về hidden chung, truyền tin, dự đoán logit cho Customer.
"""Relation-specific mean message passing with a customer classification head."""
import torch
from torch import nn


class HeteroGNN(nn.Module):
    def __init__(self, dimensions, relations, hidden=32, layers=2):
        super().__init__()
        self.relations = tuple(relations)
        # Mỗi loại node có số chiều đầu vào khác nhau nhưng đều được chiếu về hidden chung.
        self.input = nn.ModuleDict({node: nn.Linear(dim, hidden) for node, dim in dimensions.items()})
        self.self_layers = nn.ModuleList([
            nn.ModuleDict({node: nn.Linear(hidden, hidden) for node in dimensions}) for _ in range(layers)])
        # Mỗi lớp và mỗi chiều quan hệ có Linear riêng; has và rev_has không chia sẻ trọng số.
        self.messages = nn.ModuleList([
            nn.ModuleDict({relation: nn.Linear(hidden, hidden, bias=False) for relation in self.relations})
            for _ in range(layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, features, edges):
        hidden = {node: torch.relu(self.input[node](value)) for node, value in features.items()}
        for self_layer, messages in zip(self.self_layers, self.messages):
            # Nhánh self giữ thông tin node; không cần thêm cạnh self-loop vào dữ liệu graph.
            updated = {node: self_layer[node](value) for node, value in hidden.items()}
            for relation in self.relations:
                source, _, target = relation.split("__")
                edge = edges[relation]
                if edge.shape[1] == 0:
                    continue
                # Lấy biểu diễn nguồn của từng cạnh: [E, hidden].
                message = messages[relation](hidden[source][edge[0]])
                # Cộng message vào node đích bằng index_add_; nhiều cạnh cùng đích được cộng dồn.
                aggregate = torch.zeros_like(updated[target]).index_add_(0, edge[1], message)
                # Chia theo bậc đích trong riêng quan hệ này; clamp tránh chia cho 0 ở node không có cạnh.
                degree = torch.bincount(edge[1], minlength=len(aggregate)).clamp_min(1).unsqueeze(1)
                updated[target] = updated[target] + aggregate / degree
            # Chỉ cập nhật hidden sau khi xử lý mọi quan hệ: một vòng tương ứng một hop đồng bộ.
            hidden = {node: torch.relu(value) for node, value in updated.items()}
        # Trả logit [số Customer]; loss dùng BCEWithLogitsLoss, sigmoid chỉ áp dụng khi dự đoán.
        return self.head(hidden["customer"]).squeeze(1)
