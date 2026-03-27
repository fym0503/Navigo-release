import argparse
import json
from pathlib import Path

import anndata
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample an evenly spaced Navigo training subset from the full interpolation atlas."
    )
    parser.add_argument("--input-data", required=True, help="Path to the full .h5ad atlas.")
    parser.add_argument("--output-data", required=True, help="Path for the sampled subset .h5ad.")
    parser.add_argument(
        "--total-cells",
        type=int,
        default=20000,
        help="Target total number of sampled cells across all selected time points.",
    )
    parser.add_argument(
        "--num-timepoints",
        type=int,
        default=10,
        help="Number of evenly spaced time points to keep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional JSON path for subset metadata. Defaults next to output-data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if they already exist.",
    )
    return parser.parse_args()


def decode_categories(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def choose_timepoints(unique_times, num_timepoints):
    if num_timepoints < 2:
        raise ValueError("--num-timepoints must be at least 2.")
    if num_timepoints > len(unique_times):
        raise ValueError("Requested more time points than are available in the atlas.")

    chosen_idx = np.linspace(0, len(unique_times) - 1, num_timepoints, dtype=int)
    chosen_idx = np.unique(chosen_idx)
    if len(chosen_idx) != num_timepoints:
        raise ValueError("Evenly spaced time-point selection produced duplicates; reduce --num-timepoints.")
    return unique_times[chosen_idx]


def main():
    args = parse_args()
    input_path = Path(args.input_data)
    output_path = Path(args.output_data)
    summary_path = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input atlas not found: {input_path}")

    if not args.overwrite:
        for path in [output_path, summary_path]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with h5py.File(input_path, "r") as h5_file:
        time_all = h5_file["obs"]["time"][:]
        unique_times = np.unique(time_all)
        selected_times = choose_timepoints(unique_times, args.num_timepoints)

        base_cells = args.total_cells // len(selected_times)
        remainder = args.total_cells % len(selected_times)

        sampled_indices = []
        per_time_summary = []

        day_group = h5_file["obs"]["day"]
        categories = decode_categories(day_group["categories"][:])
        codes = day_group["codes"][:]

        for offset, time_value in enumerate(selected_times):
            target_count = base_cells + (1 if offset < remainder else 0)
            candidate_idx = np.flatnonzero(time_all == time_value)
            if len(candidate_idx) < target_count:
                raise ValueError(
                    f"Time point {time_value} has only {len(candidate_idx)} cells, fewer than requested {target_count}."
                )

            chosen = np.sort(rng.choice(candidate_idx, size=target_count, replace=False))
            sampled_indices.append(chosen)

            day_label = categories[int(codes[candidate_idx[0]])]
            per_time_summary.append(
                {
                    "time": float(time_value),
                    "day": day_label,
                    "available_cells": int(len(candidate_idx)),
                    "sampled_cells": int(target_count),
                }
            )

    sampled_indices = np.sort(np.concatenate(sampled_indices))

    adata = anndata.read_h5ad(input_path, backed="r")
    subset = adata[sampled_indices].to_memory()
    adata.file.close()

    subset.uns["tutorial_training_subset"] = {
        "source_dataset": str(input_path),
        "seed": int(args.seed),
        "total_cells": int(subset.n_obs),
        "num_timepoints": int(len(selected_times)),
        "selected_times": np.asarray(selected_times, dtype=np.float32),
    }
    subset.write_h5ad(output_path)

    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "source_dataset": str(input_path),
                "seed": int(args.seed),
                "total_cells": int(subset.n_obs),
                "num_timepoints": int(len(selected_times)),
                "selected_times": [float(x) for x in selected_times.tolist()],
                "per_timepoint": per_time_summary,
            },
            fp,
            indent=2,
        )

    print(f"[OK] Wrote subset: {output_path}")
    print(f"[OK] Wrote summary: {summary_path}")
    print(f"Cells: {subset.n_obs} | Genes: {subset.n_vars}")
    print("Selected time points:")
    for item in per_time_summary:
        print(
            f"  time={item['time']:>4.0f} day={item['day']:<7} "
            f"sampled={item['sampled_cells']:<5} available={item['available_cells']}"
        )


if __name__ == "__main__":
    main()
