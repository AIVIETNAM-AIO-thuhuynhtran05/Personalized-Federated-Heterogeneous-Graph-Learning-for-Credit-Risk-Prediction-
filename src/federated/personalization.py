"""Personalized FL bằng fine-tune cục bộ: lấy mô hình FedAvg đã hội tụ, mỗi client train thêm
trên dữ liệu của mình; early stopping theo AUC Val của client, trong đó mô hình FedAvg ban đầu
(epoch 0) cũng là một ứng viên, nên fine-tune không làm AUC Val kém hơn FedAvg."""
import copy
import logging
import time

from src.federated.fed_baselines import train_with_early_stopping

log = logging.getLogger(__name__)


def run_finetune(clients, global_model, mcfg: dict, ftcfg: dict):
    models, info = {}, {}
    for c in clients:
        t0 = time.time()
        m, inf = train_with_early_stopping(c, copy.deepcopy(global_model), ftcfg["lr"], mcfg["weight_decay"],
                                           ftcfg["max_epochs"], ftcfg["patience"], eval_initial=True)
        inf["train_time_sec"] = round(time.time() - t0, 1)
        models[c.cid], info[c.cid] = m, inf
        log.info("client %d fine-tune | best epoch %d (0 = giữ FedAvg) | val AUC %.5f",
                 c.cid, inf["best_epoch"], inf["best_val_auc"])
    return models, info
