"""Network visualization utilities for Navigo perturbation analysis."""

import numpy as np
import pandas as pd


def collect_edges(folder, top_tfs, fibro_genes, cardi_genes):
    """Collect regulatory edges from perturbation CSV files.

    For each TF in ``top_tfs``, reads ``<folder>/<tf>.csv`` and records edges
    to genes in ``fibro_genes``. For each fibro gene, reads its CSV and
    records edges to genes in ``cardi_genes``.

    Parameters
    ----------
    folder : Path or str
        Directory containing one CSV per gene with a ``gene_name`` column and
        a ``total_change`` column.
    top_tfs : list of str
        Transcription factors to consider as source nodes.
    fibro_genes : list of str
        Fibroblast marker genes (intermediate layer).
    cardi_genes : list of str
        Cardiomyocyte marker genes (target layer).

    Returns
    -------
    edges : list of (str, str)
    edge_colors : list of str ('red' or 'blue')
    edge_widths : list of float
    df : pd.DataFrame with columns source, target, layer, total_change
    """
    from pathlib import Path
    folder = Path(folder)
    edges, colors, widths, rows = [], [], [], []

    for tf in top_tfs:
        tf_file = folder / f'{tf}.csv'
        if not tf_file.exists():
            continue
        df = pd.read_csv(tf_file)
        for g in fibro_genes:
            hit = df[df['gene_name'] == g]
            if hit.empty:
                continue
            v = float(hit['total_change'].iloc[0])
            edges.append((tf, g))
            colors.append('red' if v > 0 else 'blue')
            widths.append(abs(v) * 1000)
            rows.append({'source': tf, 'target': g, 'layer': 'TF->Fibro', 'total_change': v})

    for f in fibro_genes:
        f_file = folder / f'{f}.csv'
        if not f_file.exists():
            continue
        df = pd.read_csv(f_file)
        for g in cardi_genes:
            hit = df[df['gene_name'] == g]
            if hit.empty:
                continue
            v = float(hit['total_change'].iloc[0])
            edges.append((f, g))
            colors.append('red' if v > 0 else 'blue')
            widths.append(abs(v) * 1000)
            rows.append({'source': f, 'target': g, 'layer': 'Fibro->Cardio', 'total_change': v})

    return edges, colors, widths, pd.DataFrame(rows)


def plot_three_layer_network(top_tfs, fibro_genes, cardi_genes, edges, edge_colors, edge_widths, out_pdf=None):
    """Draw a three-layer directed graph: TF → fibro → cardio.

    Parameters
    ----------
    top_tfs, fibro_genes, cardi_genes : list of str
        Node sets for each layer (also used to determine x-positions).
    edges : list of (str, str)
    edge_colors : list of str
    edge_widths : list of float
    out_pdf : Path or str, optional
        If given, save the figure to this file (300 dpi).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(top_tfs, node_type='TF')
    G.add_nodes_from(fibro_genes, node_type='Fibro')
    G.add_nodes_from(cardi_genes, node_type='Cardi')
    G.add_edges_from(edges)

    pos = {}
    for i, n in enumerate(top_tfs):
        pos[n] = (np.linspace(0, 1, len(top_tfs))[i], 0.8)
    for i, n in enumerate(fibro_genes):
        pos[n] = (np.linspace(0, 1, len(fibro_genes))[i], 0.5)
    for i, n in enumerate(cardi_genes):
        pos[n] = (np.linspace(0, 1, len(cardi_genes))[i], 0.2)

    tab20 = plt.cm.tab20.colors
    fig = plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(G, pos, nodelist=top_tfs, node_color=[tab20[5]], node_size=10000, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=fibro_genes, node_color=[tab20[11]], node_size=7000, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=cardi_genes, node_color=[tab20[9]], node_size=9000, alpha=0.3)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=30, arrowstyle='->', min_source_margin=40, min_target_margin=40)
    nx.draw_networkx_labels(G, pos, font_size=26)

    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    return fig
