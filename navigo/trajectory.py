"""Temporal trajectory utilities for Navigo interpolation analysis."""

import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from navigo.io import to_dense, gene_names_from_var


def collapse_msmu(msmu: np.ndarray) -> np.ndarray:
    """Sum the Ms and Mu halves of a concatenated expression matrix.

    Parameters
    ----------
    msmu : np.ndarray, shape (n_cells, 2 * n_genes)
        Concatenated Ms (spliced) and Mu (unspliced) matrix.

    Returns
    -------
    np.ndarray, shape (n_cells, n_genes)
    """
    split = msmu.shape[1] // 2
    return msmu[:, :split] + msmu[:, split:]


def get_norm_msmu(adata) -> tuple:
    """Extract and normalize the Ms/Mu concatenated matrix from an AnnData.

    Looks for (in order of preference):
    1. A pre-computed ``obsm['norm_msmu']`` array.
    2. ``layers['Ms']`` and ``layers['Mu']``.
    3. ``X`` if it has ``2 * n_vars`` columns (already concatenated).

    Parameters
    ----------
    adata : anndata.AnnData

    Returns
    -------
    msmu : np.ndarray, shape (n_cells, 2 * n_genes), dtype float32
        Min-max normalized Ms/Mu matrix.
    gene_names : np.ndarray, shape (n_genes,)
        Gene name strings.
    """
    gnames = gene_names_from_var(adata.var)

    if 'norm_msmu' in adata.obsm:
        return np.asarray(adata.obsm['norm_msmu'], dtype=np.float32), gnames

    if 'Ms' in adata.layers and 'Mu' in adata.layers:
        ms = to_dense(adata.layers['Ms']).astype(np.float32)
        mu = to_dense(adata.layers['Mu']).astype(np.float32)
        msmu = np.concatenate([ms, mu], axis=1)
        denom = msmu.max(axis=0) - msmu.min(axis=0)
        msmu = (msmu - msmu.min(axis=0)) / (denom + 1e-7)
        return msmu.astype(np.float32), gnames

    x = to_dense(adata.X).astype(np.float32)
    if x.shape[1] == adata.n_vars * 2:
        return x, gnames

    raise ValueError('Could not recover a normalized Ms/Mu matrix from the AnnData object.')


def extract_gene_expression(adata, reference_genes=None) -> tuple:
    """Extract a per-cell expression matrix, optionally aligned to reference genes.

    Handles three data layouts automatically:
    - ``obsm['norm_msmu']``: pre-computed normalized Ms+Mu
    - ``layers['Ms']`` + ``layers['Mu']``: raw layers (normalized on the fly)
    - ``X``: raw expression (used as-is)

    Parameters
    ----------
    adata : anndata.AnnData
    reference_genes : array-like of str, optional
        If given, align expression columns to this gene order. Genes absent
        from ``adata`` are filled with zeros.

    Returns
    -------
    expr : np.ndarray, shape (n_cells, n_genes or len(reference_genes))
    gene_names_out : np.ndarray, shape (n_genes or len(reference_genes),)
    """
    gnames = gene_names_from_var(adata.var)
    x = to_dense(adata.X).astype(np.float32)
    has_msmu = (
        'norm_msmu' in adata.obsm
        or ('Ms' in adata.layers and 'Mu' in adata.layers)
        or x.shape[1] == adata.n_vars * 2
    )

    if has_msmu:
        msmu, gnames = get_norm_msmu(adata)
        expr = collapse_msmu(msmu)
    else:
        expr = x
        if reference_genes is not None and expr.shape[1] == len(reference_genes) * 2:
            expr = collapse_msmu(expr)
        elif expr.shape[1] != adata.n_vars and reference_genes is None:
            raise ValueError('Could not align expression matrix to the reference gene order.')

    if reference_genes is None:
        return expr, np.asarray(gnames)

    reference_genes = np.asarray(reference_genes)
    if expr.shape[1] == len(reference_genes) and np.array_equal(gnames[: len(reference_genes)], reference_genes):
        return expr, reference_genes
    if expr.shape[1] == len(reference_genes) and len(gnames) != len(reference_genes):
        return expr, reference_genes

    gene_to_idx = {gene: idx for idx, gene in enumerate(np.asarray(gnames))}
    overlap = sum(gene in gene_to_idx for gene in reference_genes)
    if overlap == 0:
        if expr.shape[1] == len(reference_genes):
            return expr, reference_genes
        raise ValueError('Could not align expression matrix to the reference gene order.')

    aligned = np.zeros((expr.shape[0], len(reference_genes)), dtype=expr.dtype)
    for j, gene in enumerate(reference_genes):
        idx = gene_to_idx.get(gene)
        if idx is not None:
            aligned[:, j] = expr[:, idx]
    return aligned, reference_genes


def interior_test_times(all_times: np.ndarray, train_times: np.ndarray) -> list:
    """Return time points that lie strictly between two training time points.

    Parameters
    ----------
    all_times : np.ndarray
        All time points in the dataset.
    train_times : np.ndarray
        Subset used for training.

    Returns
    -------
    list of float
        Time points present in ``all_times`` but not in ``train_times``,
        with at least one training time on each side.
    """
    train_set = {float(x) for x in train_times}
    valid = []
    for t in map(float, all_times):
        if t in train_set:
            continue
        if np.any(train_times < t) and np.any(train_times > t):
            valid.append(float(t))
    return valid


def neighboring_train_times(test_time: float, train_times: np.ndarray) -> tuple:
    """Return the nearest training time points before and after a test time.

    Parameters
    ----------
    test_time : float
        The held-out time point to interpolate.
    train_times : np.ndarray
        Array of training time points.

    Returns
    -------
    (prev_train, next_train) : tuple of float
    """
    prev_train = float(train_times[train_times < test_time].max())
    next_train = float(train_times[train_times > test_time].min())
    return prev_train, next_train


def transfer_labels(
    pred_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    ref_obs: pd.DataFrame,
    columns,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Transfer cell-type labels from a reference to predicted cells via 1-NN.

    Parameters
    ----------
    pred_matrix : np.ndarray, shape (n_pred, n_features)
        Expression matrix for predicted cells.
    ref_matrix : np.ndarray, shape (n_ref, n_features)
        Expression matrix for reference cells.
    ref_obs : pd.DataFrame
        Observation metadata for reference cells (must align with ref_matrix rows).
    columns : list of str
        Columns to transfer from ``ref_obs``.
    chunk_size : int
        Process ``pred_matrix`` in chunks to limit peak memory.

    Returns
    -------
    pd.DataFrame with the transferred label columns, indexed 0..n_pred-1.
    """
    if not columns:
        return pd.DataFrame(index=np.arange(pred_matrix.shape[0]))

    nearest = np.zeros(pred_matrix.shape[0], dtype=int)
    for start in range(0, pred_matrix.shape[0], chunk_size):
        stop = min(pred_matrix.shape[0], start + chunk_size)
        dist = cdist(pred_matrix[start:stop], ref_matrix)
        nearest[start:stop] = np.argmin(dist, axis=1)
    return ref_obs.iloc[nearest][list(columns)].reset_index(drop=True)


def subset_for_time(adata, time_value: float, max_cells: int, seed: int, label: str = ''):
    """Subset an AnnData to cells at a specific time point, with optional downsampling.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have a numeric ``obs['time']`` column.
    time_value : float
        Time point to select (compared with ``np.isclose``).
    max_cells : int
        Downsample to at most this many cells if the time point is larger.
    seed : int
        Random seed for reproducible downsampling.
    label : str
        Identifier used in error messages.

    Returns
    -------
    anndata.AnnData (copy)
    """
    mask = np.isclose(adata.obs['time'].to_numpy(dtype=float), float(time_value))
    subset = adata[mask].copy()
    if subset.n_obs == 0:
        raise ValueError(f'No cells found for time={time_value}{" in " + label if label else ""}.')
    if subset.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(subset.n_obs, size=max_cells, replace=False)
        subset = subset[keep].copy()
    return subset


def sample_index_array(index_array, max_cells: int, seed: int) -> np.ndarray:
    """Reproducibly subsample an index array to at most ``max_cells`` entries.

    Parameters
    ----------
    index_array : array-like of int
    max_cells : int
    seed : int

    Returns
    -------
    np.ndarray of int, sorted ascending.
    """
    index_array = np.asarray(index_array, dtype=int)
    if len(index_array) <= max_cells:
        return index_array
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(index_array, size=max_cells, replace=False))


def _parse_numeric_token(value) -> float:
    """Extract the first numeric token from a string (e.g. 'E15' → 15.0)."""
    match = re.search(r'-?\d+(?:\.\d+)?', str(value))
    if match is None:
        raise ValueError(f'Could not parse a numeric stage value from {value!r}.')
    return float(match.group())


def prepare_time_axis(adata, day_col='day', time_col='time', prefer_existing_time=False):
    """Assign a uniform float ``obs['time']`` from day labels or an existing time column.

    Supports three input layouts:

    1. ``prefer_existing_time=True`` and ``obs[time_col]`` exists: uses the
       existing numeric column directly (e.g. a previously computed time axis).
    2. ``obs[day_col]`` exists: parses day strings to floats with
       :func:`_parse_numeric_token`, then assigns evenly-spaced model times
       (0, 0.5, 1.0, …).
    3. Only ``obs[time_col]`` exists: uses the numeric time column as-is.

    Parameters
    ----------
    adata : anndata.AnnData
    day_col : str
        Column in ``obs`` with stage labels (e.g. ``'day'``).
    time_col : str
        Column in ``obs`` with numeric time values (e.g. ``'time'``).
    prefer_existing_time : bool
        If True, prefer an existing ``obs[time_col]`` over parsing ``obs[day_col]``.

    Returns
    -------
    adata : anndata.AnnData (copy)
        With ``obs['time']`` (float model time) and ``obs['day']`` (str label) added.
    model_times : np.ndarray
        Sorted unique model time values.
    model_to_day : dict
        Mapping from model time float → original day label string.
    """
    adata = adata.copy()

    if prefer_existing_time and time_col in adata.obs.columns:
        stage_values = pd.to_numeric(adata.obs[time_col], errors='raise').astype(float)
        if day_col in adata.obs.columns:
            stage_labels = adata.obs[day_col].astype(str)
        else:
            stage_labels = stage_values.map(lambda x: f'{x:g}')
            adata.obs[day_col] = stage_labels.values
        unique_stage = np.sort(stage_values.unique())
        model_times = np.asarray(unique_stage, dtype=float)
        stage_to_model = {float(s): float(s) for s in unique_stage}
    elif day_col in adata.obs.columns:
        stage_values = adata.obs[day_col].astype(str).map(_parse_numeric_token).astype(float)
        stage_labels = adata.obs[day_col].astype(str)
        unique_stage = np.sort(stage_values.unique())
        model_times = np.arange(len(unique_stage), dtype=float) * 0.5
        stage_to_model = {float(s): float(mt) for s, mt in zip(unique_stage, model_times)}
    elif time_col in adata.obs.columns:
        stage_values = pd.to_numeric(adata.obs[time_col], errors='raise').astype(float)
        stage_labels = stage_values.map(lambda x: f'{x:g}')
        adata.obs[day_col] = stage_labels.values
        unique_stage = np.sort(stage_values.unique())
        model_times = np.asarray(unique_stage, dtype=float)
        stage_to_model = {float(s): float(s) for s in unique_stage}
    else:
        raise KeyError(f'AnnData must contain either obs[{day_col!r}] or obs[{time_col!r}].')

    stage_array = stage_values.to_numpy(dtype=float)
    stage_to_label = {}
    for stage in unique_stage:
        first_idx = int(np.flatnonzero(stage_array == stage)[0])
        stage_to_label[float(stage)] = str(stage_labels.iloc[first_idx])

    adata.obs['stage_value'] = stage_array
    adata.obs['time'] = stage_values.map(stage_to_model).to_numpy(dtype=float)
    adata.obs['day'] = stage_values.map(stage_to_label).astype(str).to_numpy()

    model_to_day = {stage_to_model[float(s)]: stage_to_label[float(s)] for s in unique_stage}
    return adata, np.asarray(model_times, dtype=float), model_to_day
