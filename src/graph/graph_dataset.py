# Đọc NPZ và trích batch theo thành phần Customer; node_id được đổi sang chỉ số cục bộ của batch.
"""Customer-component batches from one persisted client graph."""
import json
import numpy as np
import torch


def load_graph(path):
    with np.load(path, allow_pickle=False) as archive:
        # Bỏ mapping dùng để audit khỏi dữ liệu train nhằm giảm bộ nhớ sau khi đọc.
        graph = {key: archive[key] for key in archive.files if not key.startswith("mapping__")}
    graph["metadata"] = json.loads(str(graph["metadata"]))
    return graph


def component_batch(graph, customers, device="cpu"):
    node_types = [key[3:] for key in graph if key.startswith("x__")]
    selected, mappings, features = {}, {}, {}
    for node in node_types:
        # Giữ toàn bộ node thuộc các Customer được chọn để không cắt rời lịch sử khỏi khách hàng.
        selected[node] = np.flatnonzero(np.isin(graph[f"owner__{node}"], customers))
        # -1 nghĩa là node ngoài batch; node được chọn nhận chỉ số mới liên tục từ 0.
        mapping = np.full(len(graph[f"x__{node}"]), -1, dtype=np.int64)
        mapping[selected[node]] = np.arange(len(selected[node]))
        mappings[node] = mapping
        features[node] = torch.as_tensor(graph[f"x__{node}"][selected[node]], device=device)
    edges = {}
    for key, value in graph.items():
        if key.startswith("edge__"):
            relation = key[6:]
            source, _, target = relation.split("__")
            src, dst = mappings[source][value[0]], mappings[target][value[1]]
            # Chỉ giữ cạnh có cả hai đầu trong batch; src/dst lúc này là chỉ số cục bộ.
            keep = (src >= 0) & (dst >= 0)
            edges[relation] = torch.as_tensor(np.vstack([src[keep], dst[keep]]), device=device)
    # Nhãn lấy theo thứ tự Customer trong batch thực tế, có thể khác thứ tự danh sách đầu vào.
    indices = selected["customer"]
    labels = torch.as_tensor(graph["y"][indices], device=device)
    return features, edges, labels, indices
