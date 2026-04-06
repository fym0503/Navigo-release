import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPTimeGRN(nn.Module):
    def __init__(self, input_dim=30, hidden_1=100, hidden_2=100):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f"input_dim must be even, got {input_dim}")

        self.input_dim = input_dim
        self.half_dim = input_dim // 2

        self.fc1 = nn.Linear(self.half_dim, hidden_1, bias=True)
        self.fc1_t = nn.Linear(self.half_dim, hidden_1, bias=True)
        self.fc2 = nn.Linear(hidden_1, hidden_2, bias=True)
        self.fc3 = nn.Linear(hidden_2, self.half_dim, bias=True)

        self.alpha_predictor = nn.Linear(self.half_dim, self.half_dim, bias=True)
        self.beta = nn.Parameter(torch.randn(self.half_dim))
        self.gamma = nn.Parameter(torch.randn(self.half_dim))

    def forward(self, x_input):
        expected_dim = self.input_dim + 1
        if x_input.dim() != 2 or x_input.shape[1] != expected_dim:
            raise ValueError(
                f"x_input must be shaped (batch, {expected_dim}), got {tuple(x_input.shape)}"
            )

        m_s = x_input[:, : self.half_dim]
        m_u = x_input[:, self.half_dim : self.input_dim]
        t = x_input[:, -1].unsqueeze(1)

        x = self.fc1(m_s) + self.fc1_t(m_s) * t
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        alpha = torch.clamp(F.relu(self.alpha_predictor(x)), min=0)
        beta = torch.clamp(torch.sigmoid(self.beta), min=0)
        gamma = torch.clamp(torch.sigmoid(self.gamma), min=0)

        velocity_s = beta * m_u - gamma * m_s
        velocity_u = alpha - beta * m_u
        return velocity_s, velocity_u, alpha, beta, gamma

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device='cpu'):
        """Load MLPTimeGRN from a checkpoint, inferring architecture from weights.

        ``fc1`` takes only ``half_dim = input_dim // 2`` features (the Ms half),
        so ``fc1.weight.shape[1] == half_dim`` and ``input_dim = 2 * half_dim``.
        """
        state = torch.load(ckpt_path, map_location='cpu')
        half_dim  = state['fc1.weight'].shape[1]
        input_dim = 2 * half_dim
        hidden_1  = state['fc1.weight'].shape[0]
        hidden_2  = state['fc2.weight'].shape[0]
        model = cls(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)
        model.load_state_dict(state)
        return model.to(device)


class Navigo:
    def __init__(self, model=None, num_steps=1000, device="cuda"):
        self.model = model
        self.N = num_steps
        self.device = device

    def get_train_tuple_sample_flow(self, z0, z1, time, next_time):
        t = torch.rand((z1.shape[0], 1), device=self.device)
        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0

        sampled_time = time.unsqueeze(1) * t + next_time.unsqueeze(1) * (1.0 - t)
        target_s = target[:, : target.shape[1] // 2]
        target_u = target[:, target.shape[1] // 2 :]
        return z_t, sampled_time, target_s, target_u

    @torch.no_grad()
    def sample_ode_time_interval(self, z_full=None, t_start=0, t_end=1, N=None):
        if N is None:
            N = self.N

        z_full_work = copy.deepcopy(z_full).detach().to(self.device)
        batch_size = 10000

        self.model.eval()
        with torch.no_grad():
            for idx in range(0, len(z_full_work), batch_size):
                z = z_full_work[idx : min(len(z_full_work), idx + batch_size)]
                t_start_iter = t_start[idx : min(len(z_full_work), idx + batch_size)]
                t_end_iter = t_end[idx : min(len(z_full_work), idx + batch_size)]
                dt_iter = (t_end_iter - t_start_iter) / N

                for step in range(N):
                    current_t = step * (t_end_iter - t_start_iter) / N + t_start_iter
                    input_z = torch.cat([z, current_t.reshape(-1, 1)], dim=1)
                    pred_s, pred_u, _, _, _ = self.model(input_z)

                    split = z.shape[1] // 2
                    z[:, :split] = torch.clamp(
                        z[:, :split] + dt_iter.unsqueeze(1) * pred_s,
                        min=0,
                    )
                    z[:, split:] = torch.clamp(
                        z[:, split:] + dt_iter.unsqueeze(1) * pred_u,
                        min=0,
                    )

                z_full_work[idx : min(len(z_full_work), idx + batch_size)] = z

        return z_full_work.detach().cpu().numpy()

    @torch.no_grad()
    def sample_ode_time_interval_knockout(
        self,
        z_full=None,
        t_start=0,
        t_end=1,
        N=None,
        index=0,
        value_s=0,
        value_u=0,
    ):
        if N is None:
            N = self.N

        if isinstance(index, int):
            index = [index]
        if isinstance(value_s, (int, float)):
            value_s = [value_s]
        if isinstance(value_u, (int, float)):
            value_u = [value_u]

        if not (len(index) == len(value_s) == len(value_u)):
            raise ValueError("`index`, `value_s`, and `value_u` must have the same length")

        z_full_work = copy.deepcopy(z_full).detach().to(self.device)
        batch_size = 10000

        self.model.eval()
        with torch.no_grad():
            for idx in range(0, len(z_full_work), batch_size):
                z = z_full_work[idx : min(len(z_full_work), idx + batch_size)]
                t_start_iter = t_start[idx : min(len(z_full_work), idx + batch_size)]
                t_end_iter = t_end[idx : min(len(z_full_work), idx + batch_size)]
                dt_iter = (t_end_iter - t_start_iter) / N

                for step in range(N):
                    current_t = step * (t_end_iter - t_start_iter) / N + t_start_iter
                    input_z = torch.cat([z, current_t.reshape(-1, 1)], dim=1)
                    pred_s, pred_u, _, _, _ = self.model(input_z)

                    split = z.shape[1] // 2
                    z[:, :split] = torch.clamp(
                        z[:, :split] + dt_iter.unsqueeze(1) * pred_s,
                        min=0,
                    )
                    z[:, split:] = torch.clamp(
                        z[:, split:] + dt_iter.unsqueeze(1) * pred_u,
                        min=0,
                    )

                    for k in range(len(index)):
                        z[:, index[k]] = value_s[k]
                        z[:, index[k] + split] = value_u[k]

                z_full_work[idx : min(len(z_full_work), idx + batch_size)] = z

        return z_full_work.detach().cpu().numpy()
