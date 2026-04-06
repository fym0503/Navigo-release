"""Gene regulatory network (GRN) analysis functions for Navigo."""

import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_ko_responses(ko_dir, pattern='*_neg0.0x_knockout_First_heart_field.csv'):
    """Load knockout response vectors from a directory of CSVs.

    Each CSV has columns ``gene`` and ``total_change``.  Returns a normalised
    response matrix (L2-normed rows) plus the gene and target-gene lists.

    Parameters
    ----------
    ko_dir : str or Path
    pattern : str
        Glob pattern for the CSV files.

    Returns
    -------
    response_matrix : np.ndarray, shape (n_ko_genes, n_target_genes)
    gene_names : list of str
        KO gene identifiers (one per row).
    all_target_genes : list of str
        Target gene identifiers (one per column).
    """
    ko_dir = Path(ko_dir)
    ko_files = sorted(ko_dir.glob(pattern))
    if not ko_files:
        raise FileNotFoundError(f'No KO CSV files found in {ko_dir} matching {pattern}')

    response_vectors, gene_names, all_target_genes = [], [], None
    for ko_file in ko_files:
        ko_gene = ko_file.stem.replace('_neg0.0x_knockout_First_heart_field', '')
        df = pd.read_csv(ko_file).sort_values('gene').set_index('gene')
        if all_target_genes is None:
            all_target_genes = df.index.tolist()
        response_vectors.append(df['total_change'].values)
        gene_names.append(ko_gene)

    mat = np.array(response_vectors, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat, gene_names, all_target_genes


# ---------------------------------------------------------------------------
# Clustering + UMAP
# ---------------------------------------------------------------------------

def cluster_and_embed(
    ko_dir,
    n_clusters=4,
    n_pca=50,
    seed=42,
    msigdb_path=None,
    n_top_genes=200,
):
    """Cluster KO response vectors and compute UMAP embedding.

    Performs PCA → KMeans → UMAP on the L2-normalised response matrix.
    Optionally runs per-cluster pathway enrichment if ``msigdb_path``
    is provided.

    Parameters
    ----------
    ko_dir : str or Path
    n_clusters : int
    n_pca : int
    seed : int
    msigdb_path : str or Path, optional
        MSigDB JSON.  If given, Fisher's exact test enrichment is computed
        for the top downregulated genes in each cluster.
    n_top_genes : int
        Number of top downregulated genes per cluster for enrichment.

    Returns
    -------
    result : dict with keys:
        ``'gene_names'``, ``'clusters'``, ``'umap_coords'``,
        ``'distances'``  (DataFrame), ``'gene_clusters'`` (DataFrame),
        ``'response_matrix'``, ``'pca_matrix'``, ``'cluster_enrichment'`` (list of DataFrames).
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    mat, gene_names, all_target_genes = load_ko_responses(ko_dir)

    pca = PCA(n_components=min(n_pca, mat.shape[0] - 1, mat.shape[1]))
    pca_mat = pca.fit_transform(mat)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(pca_mat)

    # UMAP
    import umap as _umap
    reducer = _umap.UMAP(n_components=2, random_state=seed)
    umap_coords = reducer.fit_transform(mat)

    # Distances to cluster centres
    dists = np.zeros((len(gene_names), n_clusters))
    for i in range(n_clusters):
        dists[:, i] = np.linalg.norm(pca_mat - km.cluster_centers_[i], axis=1)

    dist_df = pd.DataFrame(dists, columns=[f'dist_to_cluster_{i}' for i in range(n_clusters)])
    dist_df.insert(0, 'gene', gene_names)
    dist_df.insert(1, 'assigned_cluster', clusters)

    gene_clusters_df = pd.DataFrame({'gene': gene_names, 'cluster': clusters})

    # Embedding CSV (for downstream plotting)
    embed_df = pd.DataFrame({
        'gene': gene_names,
        'cluster': clusters,
        'umap1': umap_coords[:, 0],
        'umap2': umap_coords[:, 1],
    })

    # Pathway enrichment per cluster
    cluster_enrichment = []
    if msigdb_path is not None:
        with open(msigdb_path) as f:
            msigdb = json.load(f)
        pathways = {pw: pw for pw in msigdb if 'GOBP' in pw}

        for c in range(n_clusters):
            mask = clusters == c
            mean_resp = mat[mask].mean(axis=0)
            top_down_idx = np.argsort(mean_resp)[:n_top_genes]
            top_down = {all_target_genes[i] for i in top_down_idx}

            rows = []
            for pw in pathways:
                pw_genes = set(msigdb[pw].get('geneSymbols', []))
                n = len(pw_genes & set(all_target_genes))
                k = len(pw_genes & top_down)
                if n < 5:
                    continue
                M = len(all_target_genes)
                _, pval = stats.fisher_exact(
                    [[k, n_top_genes - k], [n - k, M - n_top_genes - n + k]],
                    alternative='greater',
                )
                rows.append({'pathway': pw, 'overlap': k, 'pathway_size': n, 'p_value': pval})

            df_e = pd.DataFrame(rows)
            df_e = df_e[df_e['p_value'] < 0.05].sort_values('p_value')
            cluster_enrichment.append(df_e)

    return {
        'gene_names': gene_names,
        'clusters': clusters,
        'umap_coords': umap_coords,
        'distances': dist_df,
        'gene_clusters': gene_clusters_df,
        'embed': embed_df,
        'response_matrix': mat,
        'pca_matrix': pca_mat,
        'all_target_genes': all_target_genes,
        'cluster_enrichment': cluster_enrichment,
    }


# ---------------------------------------------------------------------------
# Distance analysis
# ---------------------------------------------------------------------------

def top_genes_distance_matrix(distances_df, top_n=5):
    """Build a heatmap-ready matrix of gene distances to clusters.

    For each cluster, finds the ``top_n`` closest genes and builds a
    transposed matrix suitable for ``seaborn.heatmap``.

    Parameters
    ----------
    distances_df : pd.DataFrame
        Output of :func:`cluster_and_embed` ``['distances']``.
    top_n : int

    Returns
    -------
    pd.DataFrame
        Rows = clusters, columns = genes (sorted by closest cluster).
    """
    dist_cols = [c for c in distances_df.columns if c.startswith('dist_to_cluster_')]
    gene_col = 'gene'

    # Collect unique top genes across all clusters
    all_genes = []
    for col in dist_cols:
        top = distances_df.nsmallest(top_n, col)[gene_col].tolist()
        all_genes.extend(top)
    unique_genes = list(dict.fromkeys(all_genes))

    # Build heatmap matrix
    n_clusters = len(dist_cols)
    rows = []
    for gene in unique_genes:
        r = distances_df[distances_df[gene_col] == gene]
        if len(r) == 0:
            continue
        rows.append([r[c].values[0] for c in dist_cols])

    hm = pd.DataFrame(rows, index=unique_genes,
                       columns=[f'Cluster_{i}' for i in range(n_clusters)])
    hm['_min_cluster'] = hm.idxmin(axis=1)
    hm['_min_dist'] = hm.iloc[:, :n_clusters].min(axis=1)
    hm = hm.sort_values(['_min_cluster', '_min_dist']).drop(['_min_cluster', '_min_dist'], axis=1)
    return hm.T


# ---------------------------------------------------------------------------
# CHD distribution
# ---------------------------------------------------------------------------

def chd_cluster_distribution(gene_clusters_df, classification_df, chd_col='CHD classification'):
    """Cross-tabulate CHD categories with gene clusters.

    Parameters
    ----------
    gene_clusters_df : pd.DataFrame
        Must have columns ``gene``, ``cluster``.
    classification_df : pd.DataFrame
        Index = gene names, must have column ``chd_col``.
    chd_col : str

    Returns
    -------
    pd.DataFrame
        Rows = CHD categories, columns = clusters, values = percentages.
    """
    merged = gene_clusters_df.merge(
        classification_df[[chd_col]], left_on='gene', right_index=True, how='left',
    )

    categories = ['Malformation of outflow tracts', 'Functional single ventricle',
                  'Heterotaxy', 'Obstructive lesions', 'ASD', 'VSD']

    rows = []
    for cat in categories:
        cat_genes = merged[merged[chd_col].str.contains(cat, case=False, na=False)]
        total = len(cat_genes)
        for cluster in sorted(merged['cluster'].unique()):
            count = (cat_genes['cluster'] == cluster).sum()
            rows.append({
                'category': cat, 'cluster': cluster,
                'count': count, 'total': total,
                'percentage': 100.0 * count / total if total > 0 else 0.0,
            })

    return pd.DataFrame(rows).pivot(index='category', columns='cluster', values='percentage')


# ---------------------------------------------------------------------------
# Pathway enrichment comparison
# ---------------------------------------------------------------------------

def pathway_enrichment_comparison(
    ko_dir,
    classification_df,
    msigdb_path,
    term,
    chd_col='CHD classification',
    n_bottom=200,
):
    """Compare pathway enrichment between genes with/without a CHD classification.

    For each KO gene, takes the ``n_bottom`` most downregulated targets and
    tests each GOBP pathway with Fisher's exact test.  Then compares the
    fraction of KO genes yielding significant enrichment between the two
    partitions.

    Parameters
    ----------
    ko_dir : str or Path
    classification_df : pd.DataFrame
        Index = gene names, must have ``chd_col``.
    msigdb_path : str or Path
    term : str
        Classification term to partition on (e.g. ``'Malformation of outflow tracts'``).
    chd_col : str
    n_bottom : int

    Returns
    -------
    pd.DataFrame
        Top 30 pathways by ratio difference, with columns ``pathway``,
        ``with`` count, ``without`` count, ``with_ratio``, ``without_ratio``,
        ``ratio_diff``.
    """
    ko_dir = Path(ko_dir)
    with_genes = set(classification_df[
        classification_df[chd_col].str.contains(term, case=False, na=False)
    ].index)
    without_genes = set(classification_df[
        ~classification_df[chd_col].str.contains(term, case=False, na=False)
    ].index)

    with open(msigdb_path) as f:
        msigdb = json.load(f)
    pathways = {pw: pw.replace('GOBP_', '').replace('_', ' ')
                for pw in msigdb if pw.startswith('GOBP_')}

    ko_files = sorted(ko_dir.glob('*_neg0.0x_knockout_First_heart_field.csv'))
    rows = []
    for ko_file in ko_files:
        ko_gene = ko_file.stem.replace('_neg0.0x_knockout_First_heart_field', '')
        group = 'with' if ko_gene in with_genes else ('without' if ko_gene in without_genes else None)
        if group is None:
            continue
        df = pd.read_csv(ko_file).sort_values('total_change')
        all_ko = set(df['gene'])
        bottom = set(df.head(n_bottom)['gene'])
        M, N = len(all_ko), len(bottom)
        for pw, pw_name in pathways.items():
            pw_g = set(msigdb[pw].get('geneSymbols', []))
            n = len(pw_g & all_ko)
            k = len(pw_g & bottom)
            if n and N:
                _, pval = stats.fisher_exact(
                    [[k, N - k], [n - k, M - N - n + k]], alternative='greater',
                )
                rows.append({'ko_gene': ko_gene, 'group': group, 'pathway': pw_name,
                             'overlap': k, 'pathway_size': n, 'p_value': pval})

    df_all = pd.DataFrame(rows)
    df_sig = df_all[df_all['p_value'] < 0.05]
    comp = df_sig.groupby(['pathway', 'group'])['ko_gene'].nunique().unstack(fill_value=0)
    comp['with_ratio'] = comp.get('with', 0) / max(len(with_genes), 1)
    comp['without_ratio'] = comp.get('without', 0) / max(len(without_genes), 1)
    comp['ratio_diff'] = comp['with_ratio'] - comp['without_ratio']
    return comp.sort_values('ratio_diff', ascending=False).head(30).reset_index()


# ---------------------------------------------------------------------------
# Marker change analysis
# ---------------------------------------------------------------------------

def _get_atrial_ventricular_markers(deg_df, n_markers=50, p_adj_col='pvals_adj',
                                     lfc_col='logfoldchanges', name_col='names',
                                     p_threshold=0.05):
    """Extract top atrial and ventricular markers from a DEG table."""
    filtered = deg_df[deg_df[p_adj_col] < p_threshold]
    atrial = set(filtered.nlargest(n_markers, lfc_col)[name_col])
    ventricular = set(filtered.nsmallest(n_markers, lfc_col)[name_col])
    return atrial, ventricular


def marker_change_analysis(
    ko_dir,
    deg_df,
    classification_df,
    tf_genes=None,
    n_markers=50,
    gene_col='Gene',
    type_col='classification',
):
    """Compute atrial/ventricular marker change ratios from KO responses.

    For each gene in the classification table, loads its KO response CSV and
    computes the mean absolute change in atrial vs ventricular marker genes.

    Parameters
    ----------
    ko_dir : str or Path
    deg_df : pd.DataFrame
        Atrial vs ventricular DEG table (with ``pvals_adj``, ``logfoldchanges``,
        ``names`` columns).
    classification_df : pd.DataFrame
        Must have columns ``gene_col`` and ``type_col``.
    tf_genes : set of str, optional
        Set of known TF gene names for annotation.
    n_markers : int
    gene_col, type_col : str

    Returns
    -------
    pd.DataFrame
        Columns: gene, type, is_tf, atrial_marker_change,
        ventricular_marker_change, marker_v_to_a_ratio.
    """
    ko_dir = Path(ko_dir)
    atrial, ventricular = _get_atrial_ventricular_markers(deg_df, n_markers=n_markers)
    if tf_genes is None:
        tf_genes = set()

    rows = []
    for _, row in classification_df.iterrows():
        gene = row[gene_col]
        gtype = row[type_col]
        csv_path = ko_dir / f'{gene}_neg0.0x_knockout_First_heart_field.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        a_change = df[df['gene'].isin(atrial)]['total_change'].abs().mean()
        v_change = df[df['gene'].isin(ventricular)]['total_change'].abs().mean()
        ratio = v_change / a_change if a_change > 0 else 0.0
        rows.append({
            'gene': gene, 'type': gtype, 'is_tf': gene in tf_genes,
            'atrial_marker_change': a_change,
            'ventricular_marker_change': v_change,
            'marker_v_to_a_ratio': ratio,
        })

    df_out = pd.DataFrame(rows)
    order = {'VSD_only': 0, 'ASD_only': 1, 'Both': 2, 'Other': 3}
    df_out['_order'] = df_out['type'].map(order).fillna(4)
    return df_out.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)


def marker_change_from_jacobian(
    jacobian_df,
    deg_df,
    classification_df,
    tf_genes=None,
    n_markers=50,
    gene_col='Gene',
    type_col='classification',
):
    """Same as :func:`marker_change_analysis` but using a Jacobian matrix.

    Parameters
    ----------
    jacobian_df : pd.DataFrame
        Gene × gene Jacobian matrix (columns = regulator, index = target).
    deg_df, classification_df, tf_genes, n_markers, gene_col, type_col :
        Same as :func:`marker_change_analysis`.

    Returns
    -------
    pd.DataFrame
    """
    atrial, ventricular = _get_atrial_ventricular_markers(deg_df, n_markers=n_markers)
    if tf_genes is None:
        tf_genes = set()

    rows = []
    for _, row in classification_df.iterrows():
        gene = row[gene_col]
        gtype = row[type_col]
        if gene not in jacobian_df.columns:
            continue
        col = jacobian_df[gene].abs()
        a_change = col[col.index.isin(atrial)].mean()
        v_change = col[col.index.isin(ventricular)].mean()
        ratio = v_change / a_change if a_change > 0 else 0.0
        rows.append({
            'gene': gene, 'type': gtype, 'is_tf': gene in tf_genes,
            'atrial_marker_change': a_change,
            'ventricular_marker_change': v_change,
            'marker_v_to_a_ratio': ratio,
        })

    df_out = pd.DataFrame(rows)
    order = {'VSD_only': 0, 'ASD_only': 1, 'Both': 2, 'Other': 3}
    df_out['_order'] = df_out['type'].map(order).fillna(4)
    return df_out.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction network
# ---------------------------------------------------------------------------

def interaction_network_data(ko_dir, gene_names, clusters):
    """Compute pairwise gene interaction strengths from KO responses.

    Parameters
    ----------
    ko_dir : str or Path
    gene_names : list of str
    clusters : array-like of int

    Returns
    -------
    interactions : list of ((gene_a, gene_b), strength)
        Sorted by strength descending.
    gene_total_strength : dict of {gene: float}
    cluster_interactions : np.ndarray, shape (n_clusters, n_clusters)
    """
    ko_dir = Path(ko_dir)
    n_clusters = int(np.max(clusters)) + 1
    gene_set = set(gene_names)
    gene_to_cluster = {g: int(clusters[i]) for i, g in enumerate(gene_names)}

    interaction_strengths = {}
    gene_total_strength = {g: 0.0 for g in gene_names}
    cluster_mat = np.zeros((n_clusters, n_clusters))

    for ko_file in sorted(ko_dir.glob('*_neg0.0x_knockout_First_heart_field.csv')):
        ko_gene = ko_file.stem.replace('_neg0.0x_knockout_First_heart_field', '')
        if ko_gene not in gene_set:
            continue
        df = pd.read_csv(ko_file).set_index('gene')
        for tgt in gene_names:
            if tgt in df.index and tgt != ko_gene:
                s = abs(float(df.loc[tgt, 'total_change']))
                pair = tuple(sorted([ko_gene, tgt]))
                interaction_strengths[pair] = interaction_strengths.get(pair, 0) + s
                gene_total_strength[ko_gene] += s
                gene_total_strength[tgt] += s
                c1, c2 = gene_to_cluster[ko_gene], gene_to_cluster[tgt]
                cluster_mat[c1, c2] += s
                if c1 != c2:
                    cluster_mat[c2, c1] += s

    sorted_interactions = sorted(interaction_strengths.items(), key=lambda x: x[1], reverse=True)
    return sorted_interactions, gene_total_strength, cluster_mat
