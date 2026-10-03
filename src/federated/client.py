"""Một client federated: giữ local heterogeneous graph + split Train/Val/Test riêng,
chỉ trao đổi trọng số mô hình (không gửi dữ liệu hay dự đoán thô) với server."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.graph.graph_split import CustomerSubgraphLoader
from src.graph.schema import CUSTOMER


def train_epoch(model, loader, opt, loss_fn, reg_fn=None) -> float:
    """Một epoch. `reg_fn(model)` là số hạng phạt cộng thêm vào loss (FedProx, Ditto).
    Giá trị trả về là BCE trung bình (không gồm số hạng phạt)."""
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        opt.zero_grad()
        logits = model(batch.x_dict, batch.edge_index_dict)
        bce = loss_fn(logits, batch[CUSTOMER].y)
        loss = bce if reg_fn is None else bce + reg_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += bce.item() * len(logits)
        n += len(logits)
    return total / max(n, 1)


def proximal_term(reference: torch.nn.Module, coef: float):
    """coef/2 * ||w - w_ref||^2 với w_ref cố định (bản sao tại thời điểm gọi)."""
    ref = [p.detach().clone() for p in reference.parameters()]

    def reg(model):
        return (coef / 2) * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
    return reg


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids, probs = [], []
    for batch in loader:
        probs.append(torch.sigmoid(model(batch.x_dict, batch.edge_index_dict)))
        ids.append(batch[CUSTOMER].n_id)
    return torch.cat(ids).numpy(), torch.cat(probs).numpy()


class FLClient:
    def __init__(self, cid: int, data, assign: pd.DataFrame, batch_size: int, seed: int):
        self.cid = cid
        self.data = data
        self.sk = data[CUSTOMER].sk_id_curr.numpy()
        self.y = data[CUSTOMER].y.numpy()
        pos = pd.Series(np.arange(len(self.sk)), index=self.sk)
        self.idx = {s: pos.loc[assign.loc[assign["split"] == s, "SK_ID_CURR"]].values
                    for s in ["train", "val", "test"]}
        self.train_loader = CustomerSubgraphLoader(data, self.idx["train"], batch_size, shuffle=True,
                                                   seed=seed + cid)
        self.eval_loaders = {s: CustomerSubgraphLoader(data, self.idx[s], batch_size * 4)
                             for s in ["val", "test"]}

    def n(self, split: str) -> int:
        return len(self.idx[split])

    def fit(self, model, epochs: int, lr: float, weight_decay: float, reg_fn=None, opt=None) -> float:
        """Train cục bộ. Truyền `opt` để giữ trạng thái optimizer giữa các lần gọi (Local-only,
        fine-tune); mặc định tạo optimizer mới như một client FL không lưu trạng thái giữa các round."""
        opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = 0.0
        for _ in range(epochs):
            loss = train_epoch(model, self.train_loader, opt, loss_fn, reg_fn)
        return loss

    def predict(self, model, split: str) -> pd.DataFrame:
        ids, p = predict(model, self.eval_loaders[split])
        return pd.DataFrame({"SK_ID_CURR": self.sk[ids], "TARGET": self.y[ids].astype(int),
                             "prob": p, "split": split, "client": self.cid})

    def auc(self, model, split: str = "val") -> float:
        df = self.predict(model, split)
        return float(roc_auc_score(df["TARGET"], df["prob"]))
