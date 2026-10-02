"""Local-only: mỗi client tự train mô hình riêng trên dữ liệu của mình, không trao đổi gì."""
import copy
import logging
import time

import torch

log = logging.getLogger(__name__)


def train_with_early_stopping(client, model, lr: float, weight_decay: float, max_epochs: int,
                              patience: int, eval_initial: bool = False):
    """Train trên Train của client, giữ trạng thái tốt nhất theo AUC Val của client.
    eval_initial=True: coi trạng thái ban đầu là ứng viên (dùng cho fine-tune từ mô hình chung)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    if eval_initial:
        best_auc, best_state = client.auc(model, "val"), copy.deepcopy(model.state_dict())
        history.append({"epoch": 0, "val_auc": best_auc})
    for epoch in range(1, max_epochs + 1):
        loss = client.fit(model, 1, lr, weight_decay, opt=opt)
        val_auc = client.auc(model, "val")
        history.append({"epoch": epoch, "train_loss": loss, "val_auc": val_auc})
        if val_auc > best_auc:
            best_auc, best_state, best_epoch, bad = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history}


def run_local_only(client, model_fn, mcfg: dict, lcfg: dict):
    t0 = time.time()
    model, info = train_with_early_stopping(client, model_fn(), mcfg["lr"], mcfg["weight_decay"],
                                            lcfg["max_epochs"], lcfg["patience"])
    info["train_time_sec"] = round(time.time() - t0, 1)
    log.info("client %d local-only | best epoch %d | val AUC %.5f | %.0fs",
             client.cid, info["best_epoch"], info["best_val_auc"], info["train_time_sec"])
    return model, info
