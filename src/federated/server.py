"""Vòng lặp federated dùng chung cho FedAvg, FedProx, FedPer và Ditto.

Mỗi round: server phát mô hình global -> từng client train cục bộ -> server gộp trọng số (FedAvg).
  fedavg  : client train bản sao của global.
  fedprox : như fedavg + số hạng phạt mu/2 ||w - w_global||^2 (Li et al., 2020).
  fedper  : các tham số có tiền tố trong `personal_prefixes` (head phân loại) ở lại client;
            mô hình của client = phần chung từ global + head riêng (Arivazhagan et al., 2019).
  ditto   : ngoài bản global, mỗi client giữ mô hình riêng v, train với phạt
            lam/2 ||v - w_global||^2 (Li et al., 2021).
Checkpoint chọn theo AUC Val trung bình có trọng số (theo số mẫu Val) của mô hình mà mỗi client
thực sự dùng để dự đoán; client chỉ gửi con số AUC này về server.
"""
import copy
import logging
import time

import numpy as np

from src.federated.aggregation import fedavg
from src.federated.client import proximal_term

log = logging.getLogger(__name__)


def _is_personal(key: str, prefixes) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _client_model(global_model, personal_state, algo, prefixes):
    """Mô hình client dùng để dự đoán ở thời điểm hiện tại."""
    if algo in ("fedavg", "fedprox") or personal_state is None:
        return global_model
    m = copy.deepcopy(global_model)
    if algo == "fedper":
        state = m.state_dict()
        state.update(personal_state)
        m.load_state_dict(state)
    else:  # ditto
        m.load_state_dict(personal_state)
    return m


def run_federated(clients, model_fn, mcfg: dict, acfg: dict, algo: str = "fedavg"):
    lr, wd = mcfg["lr"], mcfg["weight_decay"]
    prefixes = acfg.get("personal_prefixes", ["head."])
    global_model = model_fn()
    personal = {c.cid: None for c in clients}
    best = {"auc": -1.0, "round": 0, "global": None, "personal": None}
    bad, history = 0, []
    t0 = time.time()

    for rnd in range(1, acfg["rounds"] + 1):
        states, weights, losses = [], [], []
        for c in clients:
            local = copy.deepcopy(global_model)
            if algo == "fedper" and personal[c.cid] is not None:
                state = local.state_dict()
                state.update(personal[c.cid])
                local.load_state_dict(state)
            reg = proximal_term(global_model, acfg["mu"]) if algo == "fedprox" else None
            losses.append(c.fit(local, acfg["local_epochs"], lr, wd, reg_fn=reg))
            states.append(local.state_dict())
            weights.append(c.n("train"))

            if algo == "fedper":
                personal[c.cid] = {k: v.clone() for k, v in local.state_dict().items()
                                   if _is_personal(k, prefixes)}
            elif algo == "ditto":
                # Mô hình riêng v được kéo về global của round hiện tại (trước khi gộp)
                v = copy.deepcopy(global_model)
                if personal[c.cid] is not None:
                    v.load_state_dict(personal[c.cid])
                c.fit(v, acfg["local_epochs"], lr, wd, reg_fn=proximal_term(global_model, acfg["lam"]))
                personal[c.cid] = copy.deepcopy(v.state_dict())

        global_model.load_state_dict(fedavg(states, weights))

        aucs = [c.auc(_client_model(global_model, personal[c.cid], algo, prefixes), "val") for c in clients]
        val_auc = float(np.average(aucs, weights=[c.n("val") for c in clients]))
        history.append({"round": rnd, "client_train_loss": losses, "client_val_auc": aucs,
                        "weighted_val_auc": val_auc})
        log.info("%s round %2d | weighted val AUC %.5f | per-client %s | %.0fs", algo, rnd, val_auc,
                 np.round(aucs, 4).tolist(), time.time() - t0)

        if val_auc > best["auc"]:
            best = {"auc": val_auc, "round": rnd, "global": copy.deepcopy(global_model.state_dict()),
                    "personal": copy.deepcopy(personal)}
            bad = 0
        else:
            bad += 1
            if bad >= acfg["patience"]:
                log.info("%s early stopping (best round %d)", algo, best["round"])
                break

    global_model.load_state_dict(best["global"])
    models = {c.cid: _client_model(global_model, best["personal"][c.cid], algo, prefixes) for c in clients}
    info = {"algo": algo, "best_round": best["round"], "best_weighted_val_auc": best["auc"],
            "history": history, "train_time_sec": round(time.time() - t0, 1)}
    return global_model, models, info


def run_fedavg(clients, model_fn, mcfg: dict, fcfg: dict, seed: int = 0):
    global_model, _, info = run_federated(clients, model_fn, mcfg, fcfg, "fedavg")
    return global_model, info
