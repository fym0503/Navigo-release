import argparse
import json
from pathlib import Path

import anndata
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from navigo.distance import earth_mover_distance
from navigo.model import MLPTimeGRN, Navigo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate held-out intermediate timepoint predictions with EMD-based distance metrics."
    )
    parser.add_argument("--subset-data", required=True, help="Training subset .h5ad used for fitting.")
    parser.add_argument("--full-data", required=True, help="Full atlas .h5ad for held-out ground truth.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint from submission/main_navigo.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and summary files.")
    parser.add_argument("--hidden-1", type=int, default=5012, help="Model hidden layer width 1.")
    parser.add_argument("--hidden-2", type=int, default=5012, help="Model hidden layer width 2.")
    parser.add_argument("--flow-steps", type=int, default=10, help="ODE steps for model wrapper.")
    parser.add_argument("--integration-steps", type=int, default=25, help="ODE steps for prediction.")
    parser.add_argument(
        "--max-cells-per-group",
        type=int,
        default=300,
        help="Maximum sampled cells for each of start, prediction, ground truth, and end anchor.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device_arg


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def normalize_with_reference(ms, mu, data_min, data_max):
    stacked = np.concatenate([to_dense(ms), to_dense(mu)], axis=1).astype(np.float32, copy=False)
    denom = np.maximum(data_max - data_min, 1e-7)
    normalized = (stacked - data_min) / denom
    return np.clip(normalized, 0.0, 1.0)


def build_validation_specs(train_times, full_unique_times):
    specs = []
    full_unique_times = np.asarray(full_unique_times, dtype=np.float32)
    for idx in range(len(train_times) - 1):
        source_time = float(train_times[idx])
        next_train_time = float(train_times[idx + 1])
        gap_values = [float(x) for x in full_unique_times if source_time < float(x) < next_train_time]
        for target_time in gap_values:
            specs.append(
                {
                    "source_time": source_time,
                    "target_time": float(target_time),
                    "next_train_time": next_train_time,
                }
            )
    return specs


def sample_index(rng, index, max_cells):
    index = np.asarray(index, dtype=int)
    if len(index) <= max_cells:
        return np.sort(index)
    return np.sort(rng.choice(index, size=max_cells, replace=False))


def create_validation_plot(metrics_df, figure_path):
    ordered = metrics_df.sort_values(["target_time", "source_time"]).reset_index(drop=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [3.0, 1.35]},
    )

    x = np.arange(len(ordered))
    labels = ordered["target_day"].astype(str).tolist()

    axes[0].plot(x, ordered["prediction_emd"], marker="o", linewidth=2, color="#f58518", label="Prediction vs GT")
    axes[0].plot(x, ordered["start_emd"], marker="o", linewidth=2, color="#54a24b", label="Start vs GT")
    axes[0].plot(x, ordered["end_anchor_emd"], marker="o", linewidth=2, color="#4c78a8", label="End anchor vs GT")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("Earth mover distance")
    axes[0].set_xlabel("Held-out intermediate day")
    axes[0].set_title("Held-out intermediate EMD by timepoint")
    axes[0].legend(frameon=False, loc="best")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    summary = pd.Series(
        {
            "Prediction vs GT": float(ordered["prediction_emd"].mean()),
            "Start vs GT": float(ordered["start_emd"].mean()),
            "End anchor vs GT": float(ordered["end_anchor_emd"].mean()),
        }
    )
    colors = ["#f58518", "#54a24b", "#4c78a8"]
    axes[1].bar(summary.index, summary.values, color=colors, alpha=0.9)
    axes[1].set_ylabel("Mean earth mover distance")
    axes[1].set_title("Average EMD across held-out timepoints")
    axes[1].set_ylim(bottom=4.0)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = anndata.read_h5ad(args.subset_data)
    subset_ms = to_dense(subset.layers["Ms"])
    subset_mu = to_dense(subset.layers["Mu"])
    subset_raw = np.concatenate([subset_ms, subset_mu], axis=1).astype(np.float32, copy=False)
    data_min = subset_raw.min(axis=0)
    data_max = subset_raw.max(axis=0)
    subset_norm = np.clip((subset_raw - data_min) / np.maximum(data_max - data_min, 1e-7), 0.0, 1.0)

    train_times = np.sort(np.unique(np.asarray(subset.obs["time"], dtype=np.float32)))
    subset_days = np.asarray(subset.obs["day"]).astype(str)
    subset_time_values = np.asarray(subset.obs["time"], dtype=np.float32)

    full_adata = anndata.read_h5ad(args.full_data)
    full_times = np.asarray(full_adata.obs["time"], dtype=np.float32)
    full_days = np.asarray(full_adata.obs["day"]).astype(str)
    full_unique_times = np.sort(np.unique(full_times))

    specs = build_validation_specs(train_times, full_unique_times)
    if not specs:
        raise ValueError("No held-out intermediate time points are available for validation.")

    model = MLPTimeGRN(input_dim=subset_norm.shape[1], hidden_1=args.hidden_1, hidden_2=args.hidden_2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    navigo = Navigo(model=model, num_steps=args.flow_steps, device=device)

    metrics_rows = []
    for offset, spec in enumerate(specs):
        source_time = spec["source_time"]
        target_time = spec["target_time"]
        next_train_time = spec["next_train_time"]

        source_idx = np.flatnonzero(subset_time_values == source_time)
        gt_idx = np.flatnonzero(full_times == target_time)
        end_idx = np.flatnonzero(subset_time_values == next_train_time)
        if len(source_idx) == 0 or len(gt_idx) == 0 or len(end_idx) == 0:
            continue

        source_pick = sample_index(rng, source_idx, args.max_cells_per_group)
        gt_pick = sample_index(rng, gt_idx, args.max_cells_per_group)
        end_pick = sample_index(rng, end_idx, args.max_cells_per_group)

        source_day = subset_days[source_pick[0]]
        target_day = full_days[gt_pick[0]]
        next_train_day = subset_days[end_pick[0]]

        source_z = subset_norm[source_pick]
        end_norm = subset_norm[end_pick]
        gt_adata = full_adata[gt_pick].copy()
        gt_norm = normalize_with_reference(gt_adata.layers["Ms"], gt_adata.layers["Mu"], data_min, data_max)

        pred = navigo.sample_ode_time_interval(
            z_full=torch.tensor(source_z, dtype=torch.float32),
            t_start=torch.full((len(source_pick),), source_time, dtype=torch.float32, device=device),
            t_end=torch.full((len(source_pick),), target_time, dtype=torch.float32, device=device),
            N=args.integration_steps,
        )

        start_emd = float(earth_mover_distance(gt_norm, source_z))
        prediction_emd = float(earth_mover_distance(gt_norm, pred))
        end_anchor_emd = float(earth_mover_distance(gt_norm, end_norm))

        metrics_rows.append(
            {
                "source_time": float(source_time),
                "target_time": float(target_time),
                "next_train_time": float(next_train_time),
                "source_day": str(source_day),
                "target_day": str(target_day),
                "next_train_day": str(next_train_day),
                "start_cells": int(len(source_pick)),
                "prediction_cells": int(len(pred)),
                "ground_truth_cells": int(len(gt_norm)),
                "end_anchor_cells": int(len(end_norm)),
                "start_emd": start_emd,
                "prediction_emd": prediction_emd,
                "end_anchor_emd": end_anchor_emd,
                "prediction_vs_start_improvement": float(start_emd - prediction_emd),
                "prediction_vs_end_improvement": float(end_anchor_emd - prediction_emd),
            }
        )

    if not metrics_rows:
        raise ValueError("No valid held-out intermediate metrics could be computed.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["target_time", "source_time"]).reset_index(drop=True)
    metrics_csv = output_dir / "heldout_intermediate_metrics.csv"
    metrics_json = output_dir / "heldout_intermediate_metrics.json"
    figure_path = output_dir / "heldout_intermediate_metrics.png"

    metrics_df.to_csv(metrics_csv, index=False)
    metrics_json.write_text(json.dumps(metrics_rows, indent=2))
    create_validation_plot(metrics_df, figure_path)

    summary = {
        "checkpoint": str(args.checkpoint),
        "subset_data": str(args.subset_data),
        "full_data": str(args.full_data),
        "num_timepoints": int(len(metrics_df)),
        "mean_prediction_emd": float(metrics_df["prediction_emd"].mean()),
        "mean_start_emd": float(metrics_df["start_emd"].mean()),
        "mean_end_anchor_emd": float(metrics_df["end_anchor_emd"].mean()),
        "mean_prediction_vs_start_improvement": float(metrics_df["prediction_vs_start_improvement"].mean()),
        "mean_prediction_vs_end_improvement": float(metrics_df["prediction_vs_end_improvement"].mean()),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "figure_png": str(figure_path),
    }
    summary_path = output_dir / "heldout_intermediate_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[OK] Saved figure: {figure_path}")
    print(f"[OK] Saved metrics: {metrics_csv}")
    print(f"[OK] Saved summary: {summary_path}")
    print(
        "Mean EMD distances | "
        f"prediction={summary['mean_prediction_emd']:.4f} | "
        f"start={summary['mean_start_emd']:.4f} | "
        f"end={summary['mean_end_anchor_emd']:.4f}"
    )


if __name__ == "__main__":
    main()
