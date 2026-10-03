import torch


def fedavg(states: list[dict], weights: list[float]) -> dict:
    """FedAvg (McMahan et al., 2017): trung bình trọng số theo số mẫu Train của client."""
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        if torch.is_floating_point(states[0][key]):
            out[key] = sum(s[key] * (w / total) for s, w in zip(states, weights))
        else:
            out[key] = states[0][key].clone()
    return out
