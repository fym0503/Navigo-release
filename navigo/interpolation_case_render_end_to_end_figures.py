import argparse
import json
import re
from pathlib import Path

import anndata
import matplotlib

# matplotlib.use("Agg")  # Disabled: let the caller control the backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import sparse


DISPLAY_GENES = ["Cdkn1c", "Synpo2", "Myf6", "Tnnc2", "Myog", "Aldoa"]
PANEL_FOCUS = [
    "SMALL MOLECULE METABOLIC PROCESS",
    "SKELETAL MUSCLE CONTRACTION",
    "REGULATION OF RESPONSE TO STRESS",
    "REGULATION OF DEFENSE RESPONSE",
    "ORGANIC ACID METABOLIC PROCESS",
    "MULTICELLULAR ORGANISMAL MOVEMENT",
    "LIPID METABOLIC PROCESS",
]
TRAJECTORY_GROUP_ORDER = ["Myofibroblasts", "Muscle prog.", "Myoblasts", "Myotubes"]
TRAJECTORY_COLORS = {
    "Myofibroblasts": "#377eb8",
    "Muscle prog.": "#d8afd0",
    "Myoblasts": "#c3b1ae",
    "Myotubes": "#b8add2",
}


def parse_args():
    section_dir = Path(__file__).resolve().parent
    case_dir = section_dir / "case_Myofibroblasts"

    parser = argparse.ArgumentParser(description="Render imputation tutorial result tables and figures.")
    parser.add_argument(
        "--input_data",
        default="/workspace/fuchenghao/dynflow_codebase/dynflow_dataset/aggregated_full_hvg_4000.h5ad",
        help="Path to the full developmental AnnData used by the case scripts.",
    )
    parser.add_argument(
        "--pred_dir",
        default=str(section_dir / "outputs" / "02_myofibroblasts_end_to_end" / "00_model_inference_ckpt6"),
        help="Directory containing pred_t*_to_t*.h5ad files.",
    )
    parser.add_argument(
        "--ct_to_trajectory_json",
        default=str(section_dir / "ct_to_trajectory.json"),
        help="Cell-type to trajectory mapping JSON.",
    )
    parser.add_argument("--cell_type", default="Myofibroblasts", help="Cell type used in case filenames.")
    parser.add_argument("--target_day", type=float, default=18.25, help="Target day (E-day numeric).")
    parser.add_argument("--start_day", type=float, default=8.5, help="Timeline start day used for index mapping.")
    parser.add_argument("--step", type=float, default=0.25, help="Timeline step size for index mapping.")
    parser.add_argument(
        "--day_min",
        type=float,
        default=14.0,
        help="Lower E-day bound used in the screenshot-style trajectory panel.",
    )
    parser.add_argument(
        "--day_max",
        type=float,
        default=19.5,
        help="Upper E-day bound used in the screenshot-style trajectory panel.",
    )
    parser.add_argument(
        "--case_dir",
        default=str(case_dir),
        help="Directory containing marker/pathway CSVs from the analysis steps.",
    )
    parser.add_argument(
        "--output_root",
        default=str(section_dir / "outputs" / "03_myofibroblasts_end_to_end_notebook"),
        help="Root directory for generated tables and figures.",
    )
    return parser.parse_args()


def sanitize_cell_type(cell_type: str) -> str:
    return (
        cell_type.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "|")
        .replace(")", "|")
    )


def day_to_point(target_day: float, start_day: float, step: float) -> int:
    return int(round((target_day - start_day) / step))


def _to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _normalize_ms_mu_to_x(adata: anndata.AnnData) -> anndata.AnnData:
    adata_m = np.concatenate([_to_dense(adata.layers["Ms"]), _to_dense(adata.layers["Mu"])], axis=1)
    data = (adata_m - adata_m.min(axis=0)) / (adata_m.max(axis=0) - adata_m.min(axis=0) + 1e-7)
    n_vars = adata.n_vars
    adata.X = data[:, :n_vars] + data[:, n_vars:]
    return adata


def _gene_names_from_var(var: pd.DataFrame) -> pd.Index:
    if "gene_name" in var.columns:
        return pd.Index(var["gene_name"].astype(str))
    return pd.Index(var.index.astype(str))


def _aggregate_traj_cell_type(cell_type: str) -> str:
    if cell_type in {"Muscle progenitor cells", "Muscle progenitor cells (Prdm1+)"}:
        return "Muscle prog."
    return cell_type


def _abbreviate_pathway(pathway: str) -> str:
    x = pathway.title()
    x = x.replace("Molecule", "Mol.")
    x = x.replace("Metabolic", "Metab.")
    x = x.replace("Process", "Proc.")
    x = x.replace("Skeletal", "Skel.")
    x = x.replace("Regulation Of", "Reg.")
    x = x.replace("Response To", "Response ")
    x = x.replace("Multicellular", "Multicell.")
    x = x.replace("Organismal", "Organ.")
    return x


def build_marker_display_tables(
    input_data: Path,
    pred_dir: Path,
    cell_type: str,
    target_day: float,
    start_day: float,
    step: float,
    table_dir: Path,
):
    point = day_to_point(target_day, start_day, step)
    pred_file = pred_dir / f"pred_t{point - 1}_to_t{point}.h5ad"
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")

    full_backed = anndata.read_h5ad(input_data, backed="r")
    obs = full_backed.obs.copy()
    window_days = [target_day - 0.25, target_day, target_day + 0.25, target_day + 0.5]
    required_days = {f"E{d}" for d in window_days}

    mask = ((obs["cell_type"].astype(str) == cell_type) & (obs["day"].astype(str).isin(required_days))).values
    adata_marker = full_backed[mask].to_memory()
    adata_marker = _normalize_ms_mu_to_x(adata_marker)
    gene_names = _gene_names_from_var(adata_marker.var)
    adata_marker.var_names = gene_names
    adata_marker.var_names_make_unique()

    pred_adata = anndata.read_h5ad(pred_file)
    pred_ct = pred_adata[pred_adata.obs["predicted_cell_type"].astype(str) == cell_type].copy()
    if pred_ct.n_obs == 0:
        raise ValueError(f"No predicted cells for cell_type={cell_type} in {pred_file}")

    pred_ct.var_names = _gene_names_from_var(full_backed.var)
    pred_ct.var_names_make_unique()

    available_genes = [g for g in DISPLAY_GENES if g in set(adata_marker.var_names)]
    missing_genes = [g for g in DISPLAY_GENES if g not in set(adata_marker.var_names)]
    if not available_genes:
        raise ValueError("None of the display genes are present in the normalized AnnData.")

    expr_rows = []
    for gene in available_genes:
        idx = int(np.where(adata_marker.var_names == gene)[0][0])
        day_values = []
        for day in window_days:
            day_mask = (adata_marker.obs["day"].astype(str) == f"E{day}").values
            if int(day_mask.sum()) == 0:
                day_values.append(0.0)
            else:
                day_values.append(float(np.asarray(adata_marker[day_mask].X.mean(axis=0)).ravel()[idx]))
        pred_idx = int(np.where(pred_ct.var_names == gene)[0][0])
        pred_value = float(np.asarray(pred_ct.X.mean(axis=0)).ravel()[pred_idx])

        expr_rows.append(
            {
                "gene": gene,
                "t_minus_0.25": day_values[0],
                "t_center": day_values[1],
                "t_plus_0.25": day_values[2],
                "t_plus_0.5": day_values[3],
                "t_pred": pred_value,
            }
        )

    marker_display_df = pd.DataFrame(expr_rows)
    marker_display_csv = table_dir / "01_marker_expression_display_table.csv"
    marker_display_df.to_csv(marker_display_csv, index=False)

    noisy_expr = pd.DataFrame(
        [
            marker_display_df["t_minus_0.25"].values,
            marker_display_df["t_center"].values,
            marker_display_df["t_plus_0.25"].values,
            marker_display_df["t_plus_0.5"].values,
        ],
        index=["E18.0", "E18.25", "E18.5", "E18.75"],
        columns=available_genes,
    )
    navigo_expr = pd.DataFrame(
        [
            marker_display_df["t_minus_0.25"].values,
            marker_display_df["t_pred"].values,
            marker_display_df["t_plus_0.25"].values,
            marker_display_df["t_plus_0.5"].values,
        ],
        index=["E18.0", "E18.25", "E18.5", "E18.75"],
        columns=available_genes,
    )

    all_vals = np.concatenate([noisy_expr.values.ravel(), navigo_expr.values.ravel()])
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
    noisy_scaled = (noisy_expr - lo) / denom
    navigo_scaled = (navigo_expr - lo) / denom

    noisy_out = table_dir / "01_marker_expression_noisy_scaled.csv"
    navigo_out = table_dir / "01_marker_expression_navigo_scaled.csv"
    noisy_scaled.to_csv(noisy_out)
    navigo_scaled.to_csv(navigo_out)

    return {
        "display_table": marker_display_csv,
        "noisy_table": noisy_out,
        "navigo_table": navigo_out,
        "noisy_scaled": noisy_scaled,
        "navigo_scaled": navigo_scaled,
        "missing_genes": missing_genes,
        "available_genes": available_genes,
    }


def build_pathway_plot_table(shared_csv: Path, table_dir: Path):
    shared_df = pd.read_csv(shared_csv)
    plot_df = shared_df[shared_df["pathway"].isin(PANEL_FOCUS)].copy()
    plot_df = plot_df.set_index("pathway").reindex(PANEL_FOCUS).reset_index()
    plot_df["pathway_short"] = plot_df["pathway"].map(_abbreviate_pathway)
    plot_df["Navigo"] = plot_df["pred_E18.25_neg_log_pval"].fillna(0.0)
    plot_df["Noisy"] = plot_df["real_E18.25_neg_log_pval"].fillna(0.0)

    table_out = table_dir / "01_pathway_enrichment_panel_l_plot_table.csv"
    plot_df[["pathway", "pathway_short", "Navigo", "Noisy"]].to_csv(table_out, index=False)
    return table_out, plot_df


def build_trajectory_plot_table(
    input_data: Path,
    pred_dir: Path,
    ct_to_trajectory_json: Path,
    cell_type: str,
    day_min: float,
    day_max: float,
    table_dir: Path,
):
    full_backed = anndata.read_h5ad(input_data, backed="r")
    obs = full_backed.obs.copy()
    obs["day_num"] = [float(str(i)[1:]) for i in obs["day"]]
    all_days = np.sort(obs["day_num"].unique())
    day_to_index = {float(day): idx for idx, day in enumerate(all_days)}
    index_to_day = {idx: float(day) for day, idx in day_to_index.items()}

    with open(ct_to_trajectory_json, "r") as f:
        ct_to_traj = json.load(f)

    traj = ct_to_traj.get(cell_type)
    if not traj:
        raise ValueError(f"Cell type '{cell_type}' not found in trajectory mapping JSON.")

    traj_cts = [ct for ct, value in ct_to_traj.items() if value == traj]
    day_values = [float(day) for day in all_days if day_min <= float(day) <= day_max]

    real_rows = []
    for day in day_values:
        sub = obs[(obs["day_num"] == day) & (obs["cell_type"].astype(str).isin(traj_cts))]
        counts = sub["cell_type"].astype(str).map(_aggregate_traj_cell_type).value_counts()
        total = int(counts.sum())
        for group in TRAJECTORY_GROUP_ORDER:
            count = int(counts.get(group, 0))
            real_rows.append(
                {
                    "series": "Noisy",
                    "day_num": day,
                    "day_label": f"E{day}",
                    "group": group,
                    "count": count,
                    "ratio": count / total if total > 0 else 0.0,
                    "source": "observed",
                }
            )

    pred_rows = []
    pattern = re.compile(r"pred_t(\d+)_to_t(\d+)\.h5ad$")
    pred_by_day = {}
    for pred_file in sorted(pred_dir.glob("pred_t*_to_t*.h5ad")):
        match = pattern.match(pred_file.name)
        if not match:
            continue
        target_idx = int(match.group(2))
        target_day = index_to_day.get(target_idx)
        if target_day is None or not (day_min <= target_day <= day_max):
            continue

        adata_pred = anndata.read_h5ad(pred_file)
        pred_ct = adata_pred.obs["predicted_cell_type"].astype(str)
        counts = pred_ct[pred_ct.isin(traj_cts)].map(_aggregate_traj_cell_type).value_counts()
        total = int(counts.sum())
        day_rows = []
        for group in TRAJECTORY_GROUP_ORDER:
            count = int(counts.get(group, 0))
            day_rows.append(
                {
                    "series": "Navigo",
                    "day_num": target_day,
                    "day_label": f"E{target_day}",
                    "group": group,
                    "count": count,
                    "ratio": count / total if total > 0 else 0.0,
                    "source": pred_file.name,
                }
            )
        pred_by_day[target_day] = day_rows

    if day_values and day_values[0] not in pred_by_day:
        first_real = [row.copy() for row in real_rows if row["day_num"] == day_values[0]]
        for row in first_real:
            row["series"] = "Navigo"
            row["source"] = "seed_from_observed_day"
        pred_by_day[day_values[0]] = first_real

    for day in day_values:
        pred_rows.extend(pred_by_day.get(day, []))

    trajectory_df = pd.DataFrame(real_rows + pred_rows)
    table_out = table_dir / "01_trajectory_proportion_by_time.csv"
    trajectory_df.to_csv(table_out, index=False)

    summary_rows = []
    for day in day_values:
        point = day_to_index[day]
        summary_rows.append(
            {
                "day_num": day,
                "day_label": f"E{day}",
                "time_index": point,
                "has_prediction": day in pred_by_day,
            }
        )
    pd.DataFrame(summary_rows).to_csv(table_dir / "01_trajectory_proportion_day_index_map.csv", index=False)

    return table_out, trajectory_df


def render_panel_j(trajectory_df: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 4.9),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.12},
    )

    day_labels = list(dict.fromkeys(trajectory_df.sort_values("day_num")["day_label"]))

    for ax, series_name in zip(axes, ["Noisy", "Navigo"]):
        subset = trajectory_df[trajectory_df["series"] == series_name].copy()
        pivot = subset.pivot(index="day_label", columns="group", values="ratio").reindex(day_labels).fillna(0.0)
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))

        for group in TRAJECTORY_GROUP_ORDER:
            values = pivot[group].values
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.95,
                color=TRAJECTORY_COLORS[group],
                edgecolor="none",
                label=group if series_name == "Noisy" else None,
            )
            bottom += values

        ax.set_ylim(0, 1.0)
        ax.set_ylabel(f"{series_name}\nProportion")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [Patch(facecolor=TRAJECTORY_COLORS[group], label=group) for group in TRAJECTORY_GROUP_ORDER]
    axes[0].legend(
        handles=legend_handles,
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.02, 1.18),
        loc="lower left",
        handletextpad=0.4,
        columnspacing=1.6,
    )
    axes[0].set_title("Ground truth proportion", fontsize=13, pad=16)
    axes[1].set_xticks(np.arange(len(day_labels)))
    axes[1].set_xticklabels(day_labels, rotation=90, ha="center")

    panel_path = fig_dir / "01_trajectory_proportion_panel_j_end2end.png"
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_panel_k(marker_data: dict, fig_dir: Path):
    noisy_scaled = marker_data["noisy_scaled"]
    navigo_scaled = marker_data["navigo_scaled"]

    fig, axes = plt.subplots(2, 1, figsize=(4.9, 5.8), gridspec_kw={"hspace": 0.35})
    for ax, mat, title in [
        (axes[0], noisy_scaled, "Expression Level of\nMarker Genes"),
        (axes[1], navigo_scaled, ""),
    ]:
        heat = ax.imshow(mat.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=40, ha="right")
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        if title:
            ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(heat, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_ticks([0.5, 1.0])

    panel_path = fig_dir / "01_marker_expression_panel_k_end2end.png"
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_panel_l(plot_df: pd.DataFrame, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), facecolor="white")
    y = np.arange(len(plot_df))
    ax.barh(y + 0.18, plot_df["Navigo"], height=0.34, color="#2f69a1", label="Navigo")
    ax.barh(y - 0.18, plot_df["Noisy"], height=0.34, color="#c8cacc", label="Noisy")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["pathway_short"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"-log$_{10}(p$-value$)$", fontsize=11)
    ax.axvline(2.0, ls="--", lw=1.0, color="#8a8a8a")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    panel_path = fig_dir / "01_pathway_enrichment_panel_l_end2end.png"
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return panel_path


def render_all_panels(
    input_data,
    pred_dir,
    ct_to_trajectory_json,
    cell_type,
    target_day,
    start_day,
    step,
    case_dir,
    output_root,
    day_min=14.0,
    day_max=19.5,
):
    input_data = Path(input_data)
    pred_dir = Path(pred_dir)
    ct_to_trajectory_json = Path(ct_to_trajectory_json)
    case_dir = Path(case_dir)
    output_root = Path(output_root)
    table_dir = output_root / "01_tables"
    fig_dir = output_root / "02_figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cell_type_file = sanitize_cell_type(cell_type)
    point = day_to_point(target_day, start_day, step)
    marker_csv = case_dir / f"{cell_type_file}_marker_genes_t{point}.csv"
    shared_csv = case_dir / f"{cell_type_file}_shared_pathways.csv"
    for path in [marker_csv, shared_csv, input_data, pred_dir, ct_to_trajectory_json]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    marker_data = build_marker_display_tables(
        input_data=input_data,
        pred_dir=pred_dir,
        cell_type=cell_type,
        target_day=target_day,
        start_day=start_day,
        step=step,
        table_dir=table_dir,
    )
    pathway_table_out, plot_df = build_pathway_plot_table(shared_csv=shared_csv, table_dir=table_dir)
    trajectory_table_out, trajectory_df = build_trajectory_plot_table(
        input_data=input_data,
        pred_dir=pred_dir,
        ct_to_trajectory_json=ct_to_trajectory_json,
        cell_type=cell_type,
        day_min=day_min,
        day_max=day_max,
        table_dir=table_dir,
    )

    panel_j = render_panel_j(trajectory_df=trajectory_df, fig_dir=fig_dir)
    panel_k = render_panel_k(marker_data=marker_data, fig_dir=fig_dir)
    panel_l = render_panel_l(plot_df=plot_df, fig_dir=fig_dir)

    return {
        "table_dir": table_dir,
        "fig_dir": fig_dir,
        "paths": {
            "marker_display_table": marker_data["display_table"],
            "marker_noisy_table": marker_data["noisy_table"],
            "marker_navigo_table": marker_data["navigo_table"],
            "trajectory_table": trajectory_table_out,
            "pathway_table": pathway_table_out,
            "panel_j": panel_j,
            "panel_k": panel_k,
            "panel_l": panel_l,
        },
        "missing_display_genes": marker_data["missing_genes"],
        "available_display_genes": marker_data["available_genes"],
    }


def main():
    args = parse_args()
    summary = render_all_panels(
        input_data=args.input_data,
        pred_dir=args.pred_dir,
        ct_to_trajectory_json=args.ct_to_trajectory_json,
        cell_type=args.cell_type,
        target_day=args.target_day,
        start_day=args.start_day,
        step=args.step,
        case_dir=args.case_dir,
        output_root=args.output_root,
        day_min=args.day_min,
        day_max=args.day_max,
    )

    for key, path in summary["paths"].items():
        print(f"[OK] {key}: {path}")
    if summary["missing_display_genes"]:
        print(f"[WARN] Missing display genes: {summary['missing_display_genes']}")
    print(f"[INFO] Available display genes: {summary['available_display_genes']}")


if __name__ == "__main__":
    main()
