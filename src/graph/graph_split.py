"""Mini-batch theo khách hàng. Graph là một rừng (mỗi khách hàng là một cây riêng),
nên subgraph của một batch = toàn bộ cây con của các khách hàng trong batch, không mất cạnh nào
và không có thông tin nào đi từ khách hàng Val/Test sang khách hàng Train."""
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.graph.schema import CUSTOMER, NODE_TYPES, PARENT


def _concat_ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = int(counts.sum())
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(total) - offsets + np.repeat(starts, counts)


class CustomerSubgraphLoader:
    def __init__(self, data: HeteroData, customer_idx, batch_size: int = 1024,
                 shuffle: bool = False, seed: int = 42):
        self.data = data
        self.customer_idx = np.asarray(customer_idx)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        # CSR: con của cha p là [ptr[p], ptr[p+1]) (node con đã được sắp theo cha)
        self.ptr = {}
        for child, (parent, rel, _, _) in PARENT.items():
            src = data[parent, rel, child].edge_index[0].numpy()
            assert np.all(np.diff(src) >= 0), "Node con phải được sắp theo node cha"
            cnt = np.bincount(src, minlength=data[parent].num_nodes)
            self.ptr[child] = np.concatenate([[0], np.cumsum(cnt)])

    def __len__(self):
        return int(np.ceil(len(self.customer_idx) / self.batch_size))

    def __iter__(self):
        idx = self.rng.permutation(self.customer_idx) if self.shuffle else self.customer_idx
        for i in range(0, len(idx), self.batch_size):
            yield self.subgraph(idx[i:i + self.batch_size])

    def subgraph(self, customers: np.ndarray) -> HeteroData:
        nodes = {CUSTOMER: customers}
        local_parent = {}
        # Duyệt theo thứ tự cha trước con (customer -> bureau/prev -> events)
        for child, (parent, _, _, _) in PARENT.items():
            p = nodes[parent]
            starts, ends = self.ptr[child][p], self.ptr[child][p + 1]
            counts = ends - starts
            nodes[child] = _concat_ranges(starts, counts)
            local_parent[child] = np.repeat(np.arange(len(p)), counts)

        batch = HeteroData()
        for t in NODE_TYPES:
            batch[t].x = self.data[t].x[torch.from_numpy(nodes[t])]
        batch[CUSTOMER].y = self.data[CUSTOMER].y[torch.from_numpy(customers)]
        batch[CUSTOMER].n_id = torch.from_numpy(customers)
        for child, (parent, rel, _, _) in PARENT.items():
            src = torch.from_numpy(local_parent[child])
            dst = torch.arange(len(src))
            batch[parent, rel, child].edge_index = torch.stack([src, dst])
            batch[child, f"rev_{rel}", parent].edge_index = torch.stack([dst, src])
        return batch
