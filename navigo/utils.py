import numpy as np
import torch

from navigo.distance import earth_mover_distance


def vis_log(loss_log):
    items = []
    for key, value in loss_log.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        items.append(f"{key}: {float(value):.3f}")
    print(" | ".join(items))


def calculate_distance(x, y):
    x_square = torch.sum(x ** 2, dim=1).unsqueeze(1)
    y_square = torch.sum(y ** 2, dim=1).unsqueeze(0)
    xy = torch.mm(x, y.t())
    return torch.sqrt(torch.clamp(x_square - 2 * xy + y_square, min=0.0))


def generate_alignment_cell(data, time_label):
    del data
    alignment_cell = np.zeros((time_label.shape[0]), dtype=int)
    time_np = time_label.detach().cpu().numpy()
    time_slice = np.sort(np.unique(time_np))
    for t_idx in range(len(time_slice) - 1):
        index = np.where(time_np == time_slice[t_idx])[0]
        future_index = np.where(time_np == time_slice[t_idx + 1])[0]
        alignment_cell[index] = future_index[np.random.choice(len(future_index), len(index))]
    return alignment_cell


def matching_forward(rectified_flow, data, time_label, device):
    rectified_flow.model.eval()
    z0 = torch.tensor(data, dtype=torch.float32)

    time_np = time_label.detach().cpu().numpy()
    time_slice = np.sort(np.unique(time_np))
    if len(time_slice) < 2:
        raise ValueError("At least two unique time points are required for matching.")

    time_to_idx = {float(v): idx for idx, v in enumerate(time_slice)}
    next_time_np = np.array(
        [time_slice[min(time_to_idx[float(t)] + 1, len(time_slice) - 1)] for t in time_np],
        dtype=np.float32,
    )
    next_time = torch.tensor(next_time_np, dtype=time_label.dtype, device=device)

    pred_forward = rectified_flow.sample_ode_time_interval(
        z_full=z0,
        t_start=time_label.to(device),
        t_end=next_time,
        N=100,
    )

    alignment_cell = np.zeros((z0.shape[0]), dtype=int)
    score = {}

    for t_idx in range(len(time_slice) - 1):
        current_t = time_slice[t_idx]
        future_t = time_slice[t_idx + 1]
        current_index = np.where(time_np == current_t)[0]
        future_index = np.where(time_np == future_t)[0]

        pred_now = pred_forward[current_index]
        future_data = z0[future_index].cpu().numpy()
        current_data = z0[current_index].cpu().numpy()

        score[float(current_t)] = {
            "prediction": earth_mover_distance(future_data, pred_now),
            "baseline": earth_mover_distance(future_data, current_data),
        }

        distances = calculate_distance(
            torch.tensor(pred_now, dtype=torch.float32, device=device),
            z0[future_index].to(device),
        )
        nearest_idx = torch.argmin(distances, dim=1)
        alignment_cell[current_index] = future_index[nearest_idx.detach().cpu().numpy()]

    return alignment_cell, score


def set_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch random seeds for reproducibility."""
    import random
    import os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
