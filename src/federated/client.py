# Các hàm train và dự đoán dùng chung; chỉ train trên thành phần khách hàng trong train_mask.
"""Local optimization uses only complete train-customer components."""
import numpy as np
import torch
from src.graph.graph_dataset import component_batch


def train_local(model, graph, epochs=1, batch_size=256, learning_rate=1e-3, seed=42,
                device="cpu", optimizer=None, stats=None):
    # Caller truyền optimizer để giữ Adam state; không truyền thì khởi tạo lại cho lần gọi này.
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    # Không lấy Customer test vào batch train; node lịch sử cũng được lọc theo ownership.
    customers = np.flatnonzero(graph["train_mask"])
    model.train()
    loss_sum, examples, steps = 0.0, 0, 0
    for _ in range(epochs):
        rng.shuffle(customers)
        for start in range(0, len(customers), batch_size):
            x, edges, labels, _ = component_batch(graph, customers[start:start + batch_size], device)
            optimizer.zero_grad()
            # Loss trung bình trên Customer trong batch, không trọng số và không tính nhãn node lịch sử.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, edges), labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(labels)
            examples += len(labels)
            steps += 1
    if stats is not None:
        stats.update(train_loss=loss_sum / examples if examples else None,
                     examples=examples, optimizer_steps=steps)
    return optimizer


def predict_test(model, graph, batch_size=256, device="cpu"):
    model.eval()
    customers = np.flatnonzero(graph["test_mask"])
    labels, scores = [], []
    # Dự đoán không dựng đồ thị gradient; model.eval() không tự thay thế no_grad().
    with torch.no_grad():
        for start in range(0, len(customers), batch_size):
            x, edges, y, _ = component_batch(graph, customers[start:start + batch_size], device)
            labels.extend(y.cpu().numpy().tolist())
            scores.extend(torch.sigmoid(model(x, edges)).cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(scores)
