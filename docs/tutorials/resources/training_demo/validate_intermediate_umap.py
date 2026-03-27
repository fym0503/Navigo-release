import argparse
import json
import math
import os
from pathlib import Path

import anndata
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import umap
from scipy import sparse

from navigo.model import MLPTimeGRN, Navigo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate held-out intermediate timepoint predictions with shared-embedding UMAP overlays."
    )
    parser.add_argument("--subset-data", required=True, help="Training subset .h5ad used for fitting.")
    parser.add_argument("--full-data", required=True, help="Full atlas .h5ad for held-out ground truth.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint from submission/main_navigo.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures and summary JSON.")
    parser.add_argument("--hidden-1", type=int, default=5012, help="Model hidden layer width 1.")
    parser.add_argument("--hidden-2", type=int, default=5012, help="Model hidden layer width 2.")
    parser.add_argument("--flow-steps", type=int, default=10, help="ODE steps for model wrapper.")
    parser.add_argument("--integration-steps", type=int, default=25, help="ODE steps for prediction.")
    parser.add_argument("--cells-per-group", type=int, default=150, help="Sample size per panel and group.")
    parser.add_argument("--num-panels", type=int, default=3, help="Number of intermediate timepoints to show.")
    parser.add_argument(
        "--all-heldout",
        action="store_true",
        help="Generate UMAP panels for all held-out intermediate time points, not just a subset.",
    )
    parser.add_argument("--panels-per-row", type=int, default=5, help="Panel columns in the overview figure.")
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
        current_time = float(train_times[idx])
        next_time = float(train_times[idx + 1])
        gap_values = [float(x) for x in full_unique_times if current_time < float(x) < next_time]
        for target_time in gap_values:
            specs.append(
                {
                    "source_time": current_time,
                    "target_time": target_time,
                    "next_train_time": next_time,
                }
            )
    return specs


def choose_validation_specs(train_times, full_unique_times, num_panels):
    specs = build_validation_specs(train_times, full_unique_times)
    if not specs:
        raise ValueError("No held-out intermediate time points are available for validation.")
    if len(specs) <= num_panels:
        return specs
    chosen_idx = np.linspace(0, len(specs) - 1, num_panels, dtype=int)
    chosen_idx = np.unique(chosen_idx)
    return [specs[i] for i in chosen_idx[:num_panels]]


def plot_panel(ax, embedding, labels, source_day, target_day, next_train_day, show_legend=False):
    start_mask = labels == "start"
    pred_mask = labels == "prediction"
    gt_mask = labels == "ground_truth"
    end_mask = labels == "end_anchor"

    ax.scatter(embedding[start_mask, 0], embedding[start_mask, 1], s=10, alpha=0.40, c="#54a24b", label="Start")
    ax.scatter(embedding[gt_mask, 0], embedding[gt_mask, 1], s=10, alpha=0.40, c="#4c78a8", label="Ground truth")
    ax.scatter(embedding[pred_mask, 0], embedding[pred_mask, 1], s=10, alpha=0.40, c="#f58518", label="Prediction")
    ax.scatter(embedding[end_mask, 0], embedding[end_mask, 1], s=10, alpha=0.40, c="#b279a2", label="End anchor")
    ax.set_title(f"{source_day} -> {target_day}\n(anchor: {next_train_day})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        ax.legend(frameon=False, loc="best", fontsize=8)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    panels_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)

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

    full_backed = anndata.read_h5ad(args.full_data, backed="r")
    full_times = np.asarray(full_backed.obs["time"], dtype=np.float32)
    full_days = np.asarray(full_backed.obs["day"]).astype(str)
    full_unique_times = np.sort(np.unique(full_times))

    if args.all_heldout:
        specs = build_validation_specs(train_times, full_unique_times)
    else:
        specs = choose_validation_specs(train_times, full_unique_times, args.num_panels)
    if not specs:
        raise ValueError("No held-out intermediate time points are available for validation.")

    model = MLPTimeGRN(input_dim=subset_norm.shape[1], hidden_1=args.hidden_1, hidden_2=args.hidden_2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    navigo = Navigo(model=model, num_steps=args.flow_steps, device=device)

    panel_records = []
    combined_blocks = []

    for spec in specs:
        source_time = spec["source_time"]
        target_time = spec["target_time"]
        next_train_time = spec["next_train_time"]

        source_idx = np.flatnonzero(subset_time_values == source_time)
        gt_idx = np.flatnonzero(full_times == target_time)
        end_idx = np.flatnonzero(subset_time_values == next_train_time)
        if len(source_idx) == 0 or len(gt_idx) == 0 or len(end_idx) == 0:
            continue

        source_pick = rng.choice(source_idx, size=min(args.cells_per_group, len(source_idx)), replace=False)
        gt_pick = rng.choice(gt_idx, size=min(args.cells_per_group, len(gt_idx)), replace=False)
        end_pick = rng.choice(end_idx, size=min(args.cells_per_group, len(end_idx)), replace=False)

        source_day = subset_days[source_pick[0]]
        target_day = full_days[gt_pick[0]]
        next_train_day = subset_days[end_pick[0]]

        source_z = subset_norm[source_pick]
        pred = navigo.sample_ode_time_interval(
            z_full=torch.tensor(source_z, dtype=torch.float32),
            t_start=torch.full((len(source_pick),), source_time, dtype=torch.float32, device=device),
            t_end=torch.full((len(source_pick),), target_time, dtype=torch.float32, device=device),
            N=args.integration_steps,
        )

        gt_adata = full_backed[gt_pick].to_memory()
        gt_norm = normalize_with_reference(gt_adata.layers["Ms"], gt_adata.layers["Mu"], data_min, data_max)
        end_norm = subset_norm[end_pick]

        combined = np.concatenate([source_z, pred, gt_norm, end_norm], axis=0)
        labels = np.array(
            ["start"] * len(source_z)
            + ["prediction"] * len(pred)
            + ["ground_truth"] * len(gt_norm)
            + ["end_anchor"] * len(end_norm)
        )

        offset_start = sum(block.shape[0] for block in combined_blocks)
        combined_blocks.append(combined)
        panel_records.append(
            {
                "source_time": source_time,
                "target_time": target_time,
                "next_train_time": next_train_time,
                "next_train_day": next_train_day,
                "source_day": source_day,
                "target_day": target_day,
                "start_cells": int(len(source_z)),
                "prediction_cells": int(len(pred)),
                "ground_truth_cells": int(len(gt_norm)),
                "end_anchor_cells": int(len(end_norm)),
                "offset_start": offset_start,
                "offset_end": offset_start + combined.shape[0],
                "labels": labels.tolist(),
            }
        )

    if not panel_records:
        raise ValueError("No valid validation panels could be built from the requested specs.")

    global_data = np.concatenate(combined_blocks, axis=0)
    global_embedding = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(global_data)

    summary = []
    for idx, item in enumerate(panel_records):
        start = item["offset_start"]
        end = item["offset_end"]
        embedding = global_embedding[start:end]
        labels = np.asarray(item["labels"])

        panel_path = panels_dir / f"umap_{item['source_day']}_to_{item['target_day']}.png"
        panel_fig, panel_ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
        plot_panel(
            panel_ax,
            embedding,
            labels,
            item["source_day"],
            item["target_day"],
            item["next_train_day"],
            show_legend=(idx == 0),
        )
        panel_fig.savefig(panel_path, dpi=220, bbox_inches="tight")
        plt.close(panel_fig)

        summary.append(
            {
                "source_time": item["source_time"],
                "target_time": item["target_time"],
                "next_train_time": item["next_train_time"],
                "next_train_day": item["next_train_day"],
                "source_day": item["source_day"],
                "target_day": item["target_day"],
                "start_cells": item["start_cells"],
                "prediction_cells": item["prediction_cells"],
                "ground_truth_cells": item["ground_truth_cells"],
                "end_anchor_cells": item["end_anchor_cells"],
                "panel_png": str(panel_path),
            }
        )

    summary.sort(key=lambda x: (x["source_time"], x["target_time"]))

    ncols = min(args.panels_per_row, len(summary))
    nrows = math.ceil(len(summary) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.7 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, item in zip(axes, summary):
        ax.imshow(mpimg.imread(item["panel_png"]))
        ax.axis("off")
    for ax in axes[len(summary) :]:
        ax.axis("off")

    figure_path = output_dir / "heldout_intermediate_umap.png"
    summary_path = output_dir / "heldout_intermediate_umap.json"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"[OK] Saved figure: {figure_path}")
    print(f"[OK] Saved summary: {summary_path}")
    print(f"[OK] Panels: {len(summary)}")
    for item in summary:
        print(
            f"{item['source_day']} -> {item['target_day']} | "
            f"start={item['start_cells']} pred={item['prediction_cells']} "
            f"gt={item['ground_truth_cells']} end={item['end_anchor_cells']}"
        )

    full_backed.file.close()


if __name__ == "__main__":
    main()
