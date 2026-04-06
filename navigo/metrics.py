"""Statistical evaluation metrics for Navigo trajectory predictions."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, jensenshannon, pdist
from sklearn.decomposition import PCA


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like of float
        Raw p-values.

    Returns
    -------
    np.ndarray
        BH-adjusted p-values, clipped to [0, 1].
    """
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj_ranked[i] = val
        prev = val
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(adj_ranked, 0.0, 1.0)
    return adjusted


def wilcoxon_deg(
    x_target: np.ndarray,
    x_other: np.ndarray,
    gene_names: np.ndarray,
) -> pd.DataFrame:
    """Wilcoxon rank-sum differential expression between two cell groups.

    Parameters
    ----------
    x_target : np.ndarray, shape (n_cells_target, n_genes)
        Expression matrix for the target group.
    x_other : np.ndarray, shape (n_cells_other, n_genes)
        Expression matrix for the reference group.
    gene_names : np.ndarray, shape (n_genes,)
        Gene names corresponding to columns of the expression matrices.

    Returns
    -------
    pd.DataFrame
        Columns: ``names``, ``scores``, ``logfoldchanges``, ``pvals``,
        ``pvals_adj`` (BH-corrected). Sorted by significance then effect size.
    """
    mean_target = x_target.mean(axis=0)
    mean_other = x_other.mean(axis=0)
    logfc = np.log2((mean_target + 1e-9) / (mean_other + 1e-9))

    pvals = np.ones(x_target.shape[1], dtype=float)
    scores = np.zeros(x_target.shape[1], dtype=float)
    for j in range(x_target.shape[1]):
        a = x_target[:, j]
        b = x_other[:, j]
        if np.allclose(a, a[0]) and np.allclose(b, b[0]) and np.isclose(a[0], b[0]):
            continue
        res = stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic')
        pvals[j] = res.pvalue
        scores[j] = res.statistic

    df = pd.DataFrame({
        'names': gene_names,
        'scores': scores,
        'logfoldchanges': logfc,
        'pvals': pvals,
        'pvals_adj': bh_fdr(pvals),
    })
    return df.sort_values(
        ['pvals_adj', 'pvals', 'logfoldchanges'],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def signature_overlap(
    real_deg: pd.DataFrame,
    pred_deg: pd.DataFrame,
    direction: str,
    top_n: int = 100,
    p_adj: float = 0.05,
) -> float:
    """Jaccard-like overlap between real and predicted DEG signatures.

    Parameters
    ----------
    real_deg, pred_deg : pd.DataFrame
        DEG tables with columns ``names``, ``logfoldchanges``, ``pvals_adj``
        (as returned by :func:`wilcoxon_deg`).
    direction : {'up', 'down'}
        Whether to compare up- or down-regulated genes.
    top_n : int
        Maximum number of significant genes to consider.
    p_adj : float
        Adjusted p-value threshold for significance.

    Returns
    -------
    float
        ``|real ∩ pred| / min(|real|, |pred|)``, or ``nan`` if either set is
        empty.
    """
    if direction == 'up':
        real = real_deg[(real_deg['pvals_adj'] < p_adj) & (real_deg['logfoldchanges'] > 0)].nlargest(top_n, 'logfoldchanges')
        pred = pred_deg[(pred_deg['pvals_adj'] < p_adj) & (pred_deg['logfoldchanges'] > 0)].nlargest(top_n, 'logfoldchanges')
    elif direction == 'down':
        real = real_deg[(real_deg['pvals_adj'] < p_adj) & (real_deg['logfoldchanges'] < 0)].nsmallest(top_n, 'logfoldchanges')
        pred = pred_deg[(pred_deg['pvals_adj'] < p_adj) & (pred_deg['logfoldchanges'] < 0)].nsmallest(top_n, 'logfoldchanges')
    else:
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

    real_set = set(real['names'].astype(str))
    pred_set = set(pred['names'].astype(str))
    if not real_set or not pred_set:
        return float('nan')
    return len(real_set & pred_set) / min(len(real_set), len(pred_set))


def distribution_metrics(
    real_x: np.ndarray,
    pred_x: np.ndarray,
    seed: int = 0,
    max_cells: int = 1000,
    n_components: int = 20,
) -> dict:
    """Compute distribution-level distance metrics between real and predicted cells.

    Projects both matrices into a shared PCA space, then computes:
    - Sliced Wasserstein distance (averaged over random projections)
    - Maximum Mean Discrepancy (MMD) with RBF kernel
    - Energy distance

    Parameters
    ----------
    real_x, pred_x : np.ndarray, shape (n_cells, n_features)
        Expression matrices to compare.
    seed : int
        Random seed for subsampling and projections.
    max_cells : int
        Subsample each matrix to at most this many cells before computing.
    n_components : int
        Number of PCA components for dimensionality reduction.

    Returns
    -------
    dict with keys ``wasserstein_distance``, ``mmd``, ``energy_distance``.
    """
    rng = np.random.default_rng(seed)

    if real_x.shape[0] > max_cells:
        real_x = real_x[rng.choice(real_x.shape[0], size=max_cells, replace=False)]
    if pred_x.shape[0] > max_cells:
        pred_x = pred_x[rng.choice(pred_x.shape[0], size=max_cells, replace=False)]

    merged = np.vstack([real_x, pred_x])
    n_components = min(n_components, merged.shape[0] - 1, merged.shape[1])
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=seed)
        merged = pca.fit_transform(merged)
        real_proj = merged[: real_x.shape[0]]
        pred_proj = merged[real_x.shape[0]:]
    else:
        real_proj = real_x
        pred_proj = pred_x

    if real_proj.shape[0] < 2 or pred_proj.shape[0] < 2:
        return {'wasserstein_distance': float('nan'), 'mmd': float('nan'), 'energy_distance': float('nan')}

    n_proj = min(32, max(real_proj.shape[1], 1))
    projections = rng.normal(size=(n_proj, real_proj.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    w_dists = [stats.wasserstein_distance(real_proj @ vec, pred_proj @ vec) for vec in projections]
    wasserstein = float(np.mean(w_dists))

    pairwise = pdist(np.vstack([real_proj, pred_proj]))
    scale = float(np.median(pairwise[pairwise > 0])) if np.any(pairwise > 0) else 1.0
    gamma = 1.0 / (2.0 * scale * scale + 1e-12)

    xx = cdist(real_proj, real_proj, metric='sqeuclidean')
    yy = cdist(pred_proj, pred_proj, metric='sqeuclidean')
    xy = cdist(real_proj, pred_proj, metric='sqeuclidean')
    mmd = float(np.exp(-gamma * xx).mean() + np.exp(-gamma * yy).mean() - 2.0 * np.exp(-gamma * xy).mean())
    energy = float(2.0 * cdist(real_proj, pred_proj).mean() - cdist(real_proj, real_proj).mean() - cdist(pred_proj, pred_proj).mean())
    return {'wasserstein_distance': wasserstein, 'mmd': mmd, 'energy_distance': energy}


def cell_type_metrics(
    real_obs: pd.DataFrame,
    pred_obs: pd.DataFrame,
    real_col: str = 'cell_type',
    pred_col: str = 'predicted_cell_type',
) -> dict:
    """Compare cell-type composition between real and predicted populations.

    Parameters
    ----------
    real_obs : pd.DataFrame
        Observation metadata for the real cells.
    pred_obs : pd.DataFrame
        Observation metadata for the predicted cells.
    real_col : str
        Column in ``real_obs`` with ground-truth cell-type labels.
    pred_col : str
        Column in ``pred_obs`` with predicted cell-type labels.

    Returns
    -------
    dict with keys:
        - ``js_divergence``: Jensen-Shannon divergence between label distributions.
        - ``l1_distance``: L1 distance between label frequency vectors.
    """
    if real_col not in real_obs.columns or pred_col not in pred_obs.columns:
        return {'js_divergence': float('nan'), 'l1_distance': float('nan')}

    real_counts = real_obs[real_col].astype(str).value_counts(normalize=True)
    pred_counts = pred_obs[pred_col].astype(str).value_counts(normalize=True)
    categories = sorted(set(real_counts.index) | set(pred_counts.index))
    p = np.array([real_counts.get(cat, 0.0) for cat in categories], dtype=float)
    q = np.array([pred_counts.get(cat, 0.0) for cat in categories], dtype=float)
    return {
        'js_divergence': float(jensenshannon(p, q, base=2.0) ** 2),
        'l1_distance': float(np.abs(p - q).sum()),
    }
