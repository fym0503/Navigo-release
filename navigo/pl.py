"""Plotting utilities for Navigo (analogous to sc.pl)."""

import numpy as np
import pandas as pd


def interpolation_umap(
    umap_adata,
    focus_time=None,
    model_to_day=None,
    prev_train=None,
    next_train=None,
    out_path=None,
):
    """Plot a shared UMAP comparing interpolation methods to ground truth.

    Expects ``umap_adata`` produced by :func:`navigo.tl.build_interpolation_umap`,
    with ``obs['role']``, ``obs['method']``, and ``obsm['X_umap']``.

    Parameters
    ----------
    umap_adata : anndata.AnnData
    focus_time, model_to_day, prev_train, next_train : optional
        Override values stored in ``umap_adata.uns`` by
        :func:`navigo.tl.build_interpolation_umap`.
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import math
    import matplotlib.pyplot as plt

    focus_time = focus_time if focus_time is not None else umap_adata.uns.get('focus_time')
    model_to_day = model_to_day if model_to_day is not None else umap_adata.uns.get('model_to_day', {})
    prev_train = prev_train if prev_train is not None else umap_adata.uns.get('prev_train')
    next_train = next_train if next_train is not None else umap_adata.uns.get('next_train')

    focus_day = model_to_day.get(float(focus_time), str(focus_time)) if focus_time is not None else ''
    prev_day = model_to_day.get(float(prev_train), str(prev_train)) if prev_train is not None else 'start'
    next_day = model_to_day.get(float(next_train), str(next_train)) if next_train is not None else 'end'

    prediction_methods = [
        m for m in umap_adata.obs['method'].unique() if m not in ('observed', 'ground_truth')
    ]
    n_methods = max(len(prediction_methods), 1)
    ncols = 2 if n_methods > 1 else 1
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), squeeze=False)
    coords = umap_adata.obsm['X_umap']
    colors = {
        'observed_start': '#D8E5EA',
        'observed_end': '#C7B8E1',
        'prediction': '#BB7571',
        'ground_truth': '#5A66A0',
    }

    for panel_idx, method in enumerate(prediction_methods):
        ax = axes[panel_idx // ncols, panel_idx % ncols]
        for role, zorder, alpha in [('observed_start', 1, 0.6), ('observed_end', 2, 0.6),
                                     ('prediction', 3, 1.0), ('ground_truth', 4, 1.0)]:
            mask = umap_adata.obs['role'] == role
            if role == 'prediction':
                mask = mask & (umap_adata.obs['method'] == method)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors.get(role, '#888'), s=5, alpha=alpha, rasterized=True, zorder=zorder)
        ax.set_title(f'{method} (Interpolation Time: {focus_day})', fontsize=12)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)

    for extra_idx in range(n_methods, nrows * ncols):
        axes[extra_idx // ncols, extra_idx % ncols].axis('off')

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['observed_start'], markersize=7, label=prev_day),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['observed_end'], markersize=7, label=next_day),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['prediction'], markersize=7, label='Prediction'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['ground_truth'], markersize=7, label='Ground truth'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        from pathlib import Path
        fig.savefig(Path(out_path), dpi=200, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# General-purpose plotting helpers
# ---------------------------------------------------------------------------

def stacked_bar(
    df,
    x_col,
    group_col,
    value_col,
    group_order=None,
    colors=None,
    title=None,
    ylabel=None,
    ax=None,
    out_path=None,
):
    """Plot a single stacked bar chart of proportions.

    Parameters
    ----------
    df : pd.DataFrame
    x_col : str
        Column for the x-axis categories (e.g. ``'day_label'``).
    group_col : str
        Column whose unique values become stacked segments (e.g. ``'group'``).
    value_col : str
        Column with numeric values (e.g. ``'ratio'``).
    group_order : list of str, optional
        Stack order (bottom to top).
    colors : dict, optional
        ``{group: color}`` mapping.
    title, ylabel : str, optional
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if group_order is None:
        group_order = df[group_col].unique().tolist()
    if colors is None:
        cm = plt.colormaps.get_cmap('tab10')
        colors = {g: cm(i) for i, g in enumerate(group_order)}

    x_order = list(dict.fromkeys(df[x_col]))
    pivot = df.pivot(index=x_col, columns=group_col, values=value_col).reindex(x_order).fillna(0.0)
    x = np.arange(len(pivot))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9.5, 3.2))

    bottom = np.zeros(len(pivot))
    for group in group_order:
        vals = pivot[group].values if group in pivot.columns else np.zeros(len(pivot))
        ax.bar(x, vals, bottom=bottom, width=0.92, color=colors[group], edgecolor='none', label=group)
        bottom += vals

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(x_order, rotation=45, ha='right')
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if own_fig:
        ax.legend(ncol=min(len(group_order), 4), frameon=False, loc='upper left', bbox_to_anchor=(0, 1.18))
        plt.tight_layout()
        if out_path is not None:
            from pathlib import Path
            fig.savefig(Path(out_path), bbox_inches='tight')
        plt.show()

    return ax


def expression_heatmap(
    matrix,
    row_labels=None,
    col_labels=None,
    title=None,
    cmap='Blues',
    vmin=None,
    vmax=None,
    ax=None,
    out_path=None,
):
    """Plot a single expression heatmap.

    Parameters
    ----------
    matrix : np.ndarray or pd.DataFrame
        2-D matrix (rows = time points, columns = genes).
    row_labels, col_labels : list of str, optional
    title : str, optional
    cmap : str
    vmin, vmax : float, optional
    ax : matplotlib.axes.Axes, optional
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if isinstance(matrix, pd.DataFrame):
        if col_labels is None:
            col_labels = matrix.columns.tolist()
        if row_labels is None:
            row_labels = matrix.index.tolist()
        matrix = matrix.values

    if row_labels is None:
        row_labels = [str(i) for i in range(matrix.shape[0])]
    if col_labels is None:
        col_labels = [str(i) for i in range(matrix.shape[1])]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 0.8), max(2.5, len(row_labels) * 0.7)))

    df_plot = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    sns.heatmap(df_plot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=True)
    if title:
        ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            from pathlib import Path
            fig.savefig(Path(out_path), bbox_inches='tight')
        plt.show()

    return ax


def enrichment_barh(
    labels,
    values,
    title=None,
    color='#2f69a1',
    label=None,
    significance_line=2.0,
    ax=None,
    out_path=None,
):
    """Plot a horizontal bar chart of enrichment scores.

    Parameters
    ----------
    labels : array-like of str
        Pathway / category names.
    values : array-like of float
        -log10(p) or similar scores.
    title : str, optional
    color : str
    label : str, optional
        Legend label for this bar series.
    significance_line : float or None
    ax : matplotlib.axes.Axes, optional
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.6, max(3.0, len(labels) * 0.4)))

    y = np.arange(len(labels))
    ax.barh(y, values, height=0.7, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r'$-\log_{10}$(p-value)')
    if significance_line is not None:
        ax.axvline(significance_line, ls='--', lw=1.0, color='#8a8a8a')
    if label:
        ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            from pathlib import Path
            fig.savefig(Path(out_path), bbox_inches='tight')
        plt.show()

    return ax


def grouped_barh(
    labels,
    value_dict,
    title=None,
    significance_line=2.0,
    out_path=None,
):
    """Plot a grouped horizontal bar chart comparing multiple series.

    Parameters
    ----------
    labels : array-like of str
        Category names (y-axis).
    value_dict : dict of {str: array-like}
        ``{series_name: values}`` for each group of bars.
    title : str, optional
    significance_line : float or None
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n_series = len(value_dict)
    bar_height = 0.7 / n_series
    default_colors = ['#2f69a1', '#c8cacc', '#e07b54', '#7bc96f', '#b39bc8']

    fig, ax = plt.subplots(figsize=(7.6, max(3.0, len(labels) * 0.4)))
    y = np.arange(len(labels))

    for i, (name, vals) in enumerate(value_dict.items()):
        offset = (i - (n_series - 1) / 2) * bar_height
        color = default_colors[i % len(default_colors)]
        ax.barh(y + offset, vals, height=bar_height * 0.95, color=color, label=name)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r'$-\log_{10}$(p-value)')
    if significance_line is not None:
        ax.axvline(significance_line, ls='--', lw=1.0, color='#8a8a8a')
    ax.legend(frameon=False, ncol=n_series, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    if out_path is not None:
        from pathlib import Path
        fig.savefig(Path(out_path), bbox_inches='tight')
    plt.show()
    return fig


def marker_ratio_boxplot(
    df,
    group_col,
    value_col,
    groups=None,
    palette=None,
    title=None,
    ylabel='Impact ratio',
    ylim_max=None,
    stat_pair=None,
    ax=None,
    out_path=None,
):
    """Plot a single boxplot panel with strip overlay and optional significance bracket.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Column defining x-axis groups.
    value_col : str
        Column with numeric values for the y-axis.
    groups : list of str, optional
        Display order.
    palette : dict, optional
    title : str, optional
    ylabel : str
    ylim_max : float, optional
    stat_pair : tuple of (str, str), optional
        Two group names to compare with Mann-Whitney U test.
    ax : matplotlib.axes.Axes, optional
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if groups is None:
        groups = df[group_col].unique().tolist()
    if palette is None:
        cm = plt.colormaps.get_cmap('Set2')
        palette = {g: cm(i) for i, g in enumerate(groups)}

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(2.5, len(groups) * 1.2), 4.0))

    sns.boxplot(
        data=df, x=group_col, y=value_col,
        hue=group_col, order=groups, hue_order=groups,
        palette=palette, dodge=False, fliersize=0, linewidth=1.0, width=0.62, ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(
        data=df, x=group_col, y=value_col,
        order=groups, color='black', size=2.7, jitter=0.17, alpha=0.9, ax=ax,
    )

    if stat_pair is not None:
        from scipy.stats import mannwhitneyu
        g1, g2 = stat_pair
        v1 = df.loc[df[group_col] == g1, value_col]
        v2 = df.loc[df[group_col] == g2, value_col]
        pval = mannwhitneyu(v1, v2, alternative='two-sided').pvalue if len(v1) and len(v2) else np.nan
        idx1 = groups.index(g1) if g1 in groups else 0
        idx2 = groups.index(g2) if g2 in groups else 1
        y_max = ylim_max if ylim_max else df[value_col].max() * 1.3
        y = y_max * 0.88
        h = y_max * 0.08
        ax.plot([idx1, idx1, idx2, idx2], [y, y + h, y + h, y], color='black', lw=1)
        ax.text((idx1 + idx2) / 2, y + h + y_max * 0.02, f'P = {pval:.4f}',
                ha='center', va='bottom', fontsize=10)

    if title:
        ax.set_title(title, loc='left', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            from pathlib import Path
            fig.savefig(Path(out_path), dpi=300, bbox_inches='tight')
        plt.show()

    return ax


def perturbation_effect_plot(
    df,
    x_col,
    gene_columns,
    label_map=None,
    markers=None,
    colors=None,
    ylabel='Simulated effect',
    ylim=None,
    title=None,
    bg_color=None,
    ax=None,
    out_path=None,
):
    """Plot perturbation effects of one gene family across lineages.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``x_col`` and all columns listed in ``gene_columns``.
    x_col : str
        Column with lineage / cell-type identifiers.
    gene_columns : list of str
        Column names to plot (one line per gene).
    label_map : dict, optional
        Maps ``x_col`` values to display labels.
    markers : list of str, optional
        Matplotlib marker per gene column (e.g. ``['o', 'x', 'o']``).
    colors : list of str, optional
        Color per gene column.
    ylabel : str
    ylim : tuple, optional
    title : str, optional
    bg_color : str, optional
    ax : matplotlib.axes.Axes, optional
    out_path : str or Path, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.5), 4.0))

    if label_map is None:
        label_map = {v: v for v in df[x_col]}

    default_colors = ['#2ca02c', '#2166ac', '#ff6f0e', '#7e56b4', '#d62728']
    default_markers = ['o', 'x', 'o', 'x', 'o']
    if colors is None:
        colors = default_colors
    if markers is None:
        markers = default_markers

    x = np.arange(len(df))
    for j, col in enumerate(gene_columns):
        m = markers[j % len(markers)]
        c = colors[j % len(colors)]
        mfc = 'none' if m == 'o' else None
        ax.plot(x, df[col].values, linestyle='None', marker=m, markersize=8,
                markerfacecolor=mfc, markeredgewidth=1.5, markeredgecolor=c, color=c,
                label=col.replace('_normalized', ''))

    for xi in x:
        ax.axvline(xi, color='k', lw=1.2, ls=(0, (6, 8)), alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(v, v) for v in df[x_col]], rotation=45, ha='right', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=18)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=14)
    if bg_color:
        ax.set_facecolor(bg_color)
    ax.tick_params(axis='both', labelsize=14, width=1.0, length=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    if own_fig:
        plt.tight_layout()
        if out_path is not None:
            from pathlib import Path
            fig.savefig(Path(out_path), dpi=300, bbox_inches='tight')
        plt.show()

    return ax
