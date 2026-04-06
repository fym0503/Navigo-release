"""Analysis tools for Navigo (analogous to sc.tl)."""

import numpy as np
import pandas as pd
from navigo.trajectory import interior_test_times, neighboring_train_times  # noqa: F401


# ---------------------------------------------------------------------------
# Interpolation benchmarking
# ---------------------------------------------------------------------------

def interpolate_atlas(
    flow,
    atlas,
    train_times,
    test_times,
    infer_mode='pred_average',
    alpha=0.5,
    max_score_cells=512,
    seed=42,
):
    """Run Navigo ODE interpolation across held-out test times.

    For each held-out test time, samples source and destination cells from the
    flanking training time points, integrates the ODE forward and backward, and
    blends the predictions according to ``infer_mode``.

    Parameters
    ----------
    flow : navigo.model.Navigo
        Initialised Navigo flow model.
    atlas : anndata.AnnData
        Atlas with ``obs['time']`` (float model time) and ``obsm['norm_msmu']``.
        May be backed (memory-efficient row access is handled automatically).
        If loaded via :func:`navigo.pp.load_atlas`, ``uns`` should contain
        ``'gene_names'`` and ``'model_to_day'``.
    train_times : array-like of float
        Time points used for training.
    test_times : array-like of float
        Held-out time points to interpolate.
    infer_mode : {'pred_average', 'forward', 'backward', 'gt_average'}
        Blending strategy.  ``'pred_average'`` (default) blends ODE-forward
        and ODE-backward trajectories weighted by ``alpha``.
    alpha : float
        Weight for the forward prediction in ``'pred_average'`` / ``'gt_average'``
        modes.  ``1.0`` uses only the forward prediction.
    max_score_cells : int
        Maximum cells per group sampled for EMD scoring.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pred_adata : anndata.AnnData
        Predicted cells with ``obs['time']``, ``obs['day']``, ``obs['source_time']``,
        ``obs['dest_time']``, and ``obsm['norm_msmu']``.
    score_df : pd.DataFrame
        EMD scores per test time (columns: time, day, source_time, dest_time,
        emd_prediction, emd_baseline_source, emd_future_stage).

    Examples
    --------
    >>> pred_adata, score_df = navigo.tl.interpolate_atlas(
    ...     flow, atlas, train_times=train_times, test_times=test_times,
    ...     infer_mode='pred_average', alpha=0.5, seed=SEED,
    ... )
    """
    import torch
    import anndata as _ad
    from navigo.distance import earth_mover_distance
    from navigo.io import gene_names_from_var
    from navigo.trajectory import (
        neighboring_train_times,
        sample_index_array,
        collapse_msmu,
    )
    from navigo.utils import calculate_distance

    device = flow.device
    train_times = np.asarray(train_times, dtype=float)
    test_times = list(test_times)

    time_values = atlas.obs['time'].to_numpy(dtype=float)
    atlas_var = atlas.var.copy()
    gene_names = atlas.uns.get('gene_names', gene_names_from_var(atlas_var))
    model_to_day = atlas.uns.get('model_to_day', {})
    atlas_msmu_dim = atlas.obsm['norm_msmu'].shape[1]

    def _load_msmu_rows(row_idx):
        row_idx = np.asarray(row_idx, dtype=int)
        if row_idx.size == 0:
            return np.zeros((0, atlas_msmu_dim), dtype=np.float32)
        order = np.argsort(row_idx, kind='stable')
        sorted_idx = row_idx[order]
        unique_idx, inverse = np.unique(sorted_idx, return_inverse=True)
        rows = np.asarray(atlas.obsm['norm_msmu'][unique_idx], dtype=np.float32)[inverse]
        restored = np.empty_like(rows)
        restored[order] = rows
        return restored

    def _load_expr_rows(row_idx):
        return collapse_msmu(_load_msmu_rows(row_idx))

    pred_blocks, pred_expr_blocks, pred_obs_frames, score_rows = [], [], [], []

    for offset, test_time in enumerate(test_times):
        prev_train, next_train = neighboring_train_times(test_time, train_times)
        source_idx = np.flatnonzero(np.isclose(time_values, prev_train))
        dest_idx = np.flatnonzero(np.isclose(time_values, next_train))
        target_idx = np.flatnonzero(np.isclose(time_values, test_time))
        if len(source_idx) == 0 or len(dest_idx) == 0 or len(target_idx) == 0:
            continue

        rng = np.random.default_rng(seed + offset)
        sample_size = len(target_idx)
        source_sample_idx = rng.choice(source_idx, size=sample_size, replace=True)
        dest_sample_idx = rng.choice(dest_idx, size=sample_size, replace=True)

        z0 = torch.tensor(_load_msmu_rows(source_sample_idx), dtype=torch.float32)
        z1 = torch.tensor(_load_msmu_rows(dest_sample_idx), dtype=torch.float32)
        t0 = torch.full((sample_size,), float(prev_train), dtype=torch.float32, device=device)
        t1 = torch.full((sample_size,), float(next_train), dtype=torch.float32, device=device)
        ti = torch.full((sample_size,), float(test_time), dtype=torch.float32, device=device)

        pred_forward = flow.sample_ode_time_interval(z_full=z0, t_start=t0, t_end=t1, N=flow.N)
        pred_forward_half = flow.sample_ode_time_interval(z_full=z0, t_start=t0, t_end=ti, N=flow.N)
        pred_backward_half = flow.sample_ode_time_interval(z_full=z1, t_start=t1, t_end=ti, N=flow.N)

        distances = calculate_distance(
            torch.tensor(pred_forward, dtype=torch.float32, device=device),
            z1.to(device),
        )
        alignment = torch.argmin(distances, dim=1).detach().cpu().numpy()

        if infer_mode == 'forward':
            pred_mid = pred_forward_half
        elif infer_mode == 'backward':
            pred_mid = pred_backward_half
        elif infer_mode == 'pred_average':
            pred_mid = (1.0 - alpha) * pred_backward_half[alignment] + alpha * pred_forward_half
        elif infer_mode == 'gt_average':
            pred_mid = (1.0 - alpha) * z1.detach().cpu().numpy()[alignment] + alpha * z0.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupported infer_mode: {infer_mode!r}')

        pred_expr = collapse_msmu(pred_mid)
        s_tgt = sample_index_array(target_idx, max_cells=max_score_cells, seed=seed + 1000 + offset)
        s_src = sample_index_array(source_idx, max_cells=max_score_cells, seed=seed + 2000 + offset)
        s_dst = sample_index_array(dest_idx, max_cells=max_score_cells, seed=seed + 3000 + offset)
        s_pred = sample_index_array(np.arange(pred_expr.shape[0]), max_cells=max_score_cells, seed=seed + 4000 + offset)

        score_rows.append({
            'time': float(test_time),
            'day': model_to_day.get(float(test_time), str(test_time)),
            'source_time': float(prev_train),
            'dest_time': float(next_train),
            'n_pred_cells': int(sample_size),
            'score_sample_cells': int(len(s_pred)),
            'emd_prediction': float(earth_mover_distance(_load_expr_rows(s_tgt), pred_expr[s_pred])),
            'emd_baseline_source': float(earth_mover_distance(_load_expr_rows(s_tgt), _load_expr_rows(s_src))),
            'emd_future_stage': float(earth_mover_distance(_load_expr_rows(s_tgt), _load_expr_rows(s_dst))),
        })

        obs_frame = pd.DataFrame({
            'time': np.repeat(float(test_time), sample_size),
            'day': np.repeat(model_to_day.get(float(test_time), str(test_time)), sample_size),
            'source_time': np.repeat(float(prev_train), sample_size),
            'source_day': np.repeat(model_to_day.get(float(prev_train), str(prev_train)), sample_size),
            'dest_time': np.repeat(float(next_train), sample_size),
            'dest_day': np.repeat(model_to_day.get(float(next_train), str(next_train)), sample_size),
        })
        pred_blocks.append(pred_mid.astype(np.float32))
        pred_expr_blocks.append(pred_expr.astype(np.float32))
        pred_obs_frames.append(obs_frame)

    if not pred_blocks:
        raise RuntimeError(
            'No held-out interpolation predictions were generated. '
            'Check that test_times lie strictly between train_times in the atlas.'
        )

    pred_obs = pd.concat(pred_obs_frames, ignore_index=True)
    pred_obs.index = [f'pred_{i}' for i in range(len(pred_obs))]

    var_copy = atlas_var.copy()
    var_copy.index = pd.Index(gene_names, dtype=str)
    var_copy.index.name = None

    pred_adata = _ad.AnnData(X=np.vstack(pred_expr_blocks), obs=pred_obs, var=var_copy)
    pred_adata.obsm['norm_msmu'] = np.vstack(pred_blocks)
    pred_adata.uns['inference_mode'] = infer_mode
    pred_adata.uns['alpha'] = alpha
    pred_adata.uns['model_to_day'] = model_to_day

    score_df = pd.DataFrame(score_rows).sort_values('time').reset_index(drop=True)
    return pred_adata, score_df


def evaluate_interpolation(
    pred_adata,
    atlas,
    test_times,
    train_times,
    max_cells=1024,
    seed=42,
):
    """Quantify interpolation quality via DEG overlap, cell-type, and distribution metrics.

    Compares predicted cells at each held-out test time to real cells from the
    atlas using Wilcoxon DEG signature overlap, cell-type JS-divergence, and
    distribution distance (Wasserstein, MMD, energy distance).

    Parameters
    ----------
    pred_adata : anndata.AnnData
        Output of :func:`interpolate_atlas`.
    atlas : anndata.AnnData
        Source atlas with ``obs['time']`` and ``obsm['norm_msmu']``.
    test_times : list of float
    train_times : array-like of float
    max_cells : int
        Maximum cells sampled per group for metric computation.
    seed : int

    Returns
    -------
    pd.DataFrame
        One row per test time with columns: time, day, cell-type JS-div / L1,
        signature overlap (up/down), distribution distances, and cell counts.

    Examples
    --------
    >>> metrics_df = navigo.tl.evaluate_interpolation(
    ...     pred_adata, atlas, test_times=test_times, train_times=train_times, seed=SEED,
    ... )
    """
    from navigo.io import gene_names_from_var
    from navigo.metrics import wilcoxon_deg, signature_overlap, distribution_metrics, cell_type_metrics
    from navigo.trajectory import neighboring_train_times, sample_index_array, collapse_msmu

    train_times = np.asarray(train_times, dtype=float)
    time_values = atlas.obs['time'].to_numpy(dtype=float)
    atlas_obs = atlas.obs.copy()
    gene_names = atlas.uns.get('gene_names', gene_names_from_var(atlas.var))
    model_to_day = atlas.uns.get('model_to_day', {})
    atlas_msmu_dim = atlas.obsm['norm_msmu'].shape[1]
    pred_time_values = pred_adata.obs['time'].to_numpy(dtype=float)

    def _load_msmu_rows(row_idx):
        row_idx = np.asarray(row_idx, dtype=int)
        if row_idx.size == 0:
            return np.zeros((0, atlas_msmu_dim), dtype=np.float32)
        order = np.argsort(row_idx, kind='stable')
        sorted_idx = row_idx[order]
        unique_idx, inverse = np.unique(sorted_idx, return_inverse=True)
        rows = np.asarray(atlas.obsm['norm_msmu'][unique_idx], dtype=np.float32)[inverse]
        restored = np.empty_like(rows)
        restored[order] = rows
        return restored

    def _load_expr_rows(row_idx):
        return collapse_msmu(_load_msmu_rows(row_idx))

    metrics_rows = []
    for offset, test_time in enumerate(test_times):
        prev_train, next_train = neighboring_train_times(test_time, train_times)
        real_window_idx = np.flatnonzero(
            np.isclose(time_values, prev_train)
            | np.isclose(time_values, test_time)
            | np.isclose(time_values, next_train)
        )
        real_target_idx = np.flatnonzero(np.isclose(time_values, test_time))
        real_other_idx = real_window_idx[~np.isin(real_window_idx, real_target_idx)]
        pred_target_idx = np.flatnonzero(np.isclose(pred_time_values, test_time))

        if real_target_idx.size == 0 or real_other_idx.size == 0 or pred_target_idx.size == 0:
            continue

        s_tgt = sample_index_array(real_target_idx, max_cells=max_cells, seed=seed + 5000 + offset)
        s_oth = sample_index_array(real_other_idx, max_cells=max_cells, seed=seed + 6000 + offset)
        s_pred = sample_index_array(pred_target_idx, max_cells=max_cells, seed=seed + 7000 + offset)

        real_target_expr = _load_expr_rows(s_tgt)
        real_other_expr = _load_expr_rows(s_oth)
        pred_target_expr = np.asarray(pred_adata.X[s_pred], dtype=np.float32)

        real_deg = wilcoxon_deg(real_target_expr, real_other_expr, gene_names)
        pred_deg = wilcoxon_deg(pred_target_expr, real_other_expr, gene_names)
        ct_stats = cell_type_metrics(atlas_obs.iloc[real_target_idx], pred_adata.obs.iloc[pred_target_idx])
        dist_stats = distribution_metrics(real_target_expr, pred_target_expr, seed=seed + offset)

        metrics_rows.append({
            'time': float(test_time),
            'day': model_to_day.get(float(test_time), str(test_time)),
            **ct_stats,
            'overlap_up': signature_overlap(real_deg, pred_deg, 'up'),
            'overlap_down': signature_overlap(real_deg, pred_deg, 'down'),
            **dist_stats,
            'n_real_cells': int(real_target_idx.size),
            'n_pred_cells': int(pred_target_idx.size),
            'deg_sample_cells': int(len(s_tgt)),
        })

    return pd.DataFrame(metrics_rows).sort_values('time').reset_index(drop=True)


def build_interpolation_umap(
    pred_adata,
    atlas,
    focus_time,
    train_times,
    test_times,
    max_cells_per_group=3000,
    seed=42,
    optional_methods=None,
):
    """Assemble a shared UMAP comparing Navigo predictions to ground truth and baselines.

    Loads atlas cells at the focus time and its two flanking training times,
    appends the Navigo predictions (and any optional baseline methods), computes
    PCA + neighbors + UMAP jointly, and returns an AnnData ready for plotting.

    Parameters
    ----------
    pred_adata : anndata.AnnData
        Output of :func:`interpolate_atlas`.
    atlas : anndata.AnnData
        Source atlas (may be backed).
    focus_time : float
        The held-out test time to visualise (must be in ``test_times``).
    train_times : array-like of float
    test_times : list of float
    max_cells_per_group : int
    seed : int
    optional_methods : dict of {str: anndata.AnnData}, optional
        Additional baseline methods keyed by name.

    Returns
    -------
    umap_adata : anndata.AnnData
        With ``obs['role']``, ``obs['method']``, and ``obsm['X_umap']``.

    Examples
    --------
    >>> umap_adata = navigo.tl.build_interpolation_umap(
    ...     pred_adata, atlas, focus_time=FOCUS_TIME,
    ...     train_times=train_times, test_times=test_times, seed=SEED,
    ... )
    """
    import anndata as _ad
    import scanpy as sc
    from navigo.io import gene_names_from_var
    from navigo.trajectory import (
        neighboring_train_times,
        sample_index_array,
        collapse_msmu,
        subset_for_time,
        extract_gene_expression,
    )

    train_times = np.asarray(train_times, dtype=float)
    time_values = atlas.obs['time'].to_numpy(dtype=float)
    atlas_var = atlas.var.copy()
    gene_names = atlas.uns.get('gene_names', gene_names_from_var(atlas_var))
    model_to_day = atlas.uns.get('model_to_day', {})
    atlas_msmu_dim = atlas.obsm['norm_msmu'].shape[1]

    def _load_msmu_rows(row_idx):
        row_idx = np.asarray(row_idx, dtype=int)
        if row_idx.size == 0:
            return np.zeros((0, atlas_msmu_dim), dtype=np.float32)
        order = np.argsort(row_idx, kind='stable')
        sorted_idx = row_idx[order]
        unique_idx, inverse = np.unique(sorted_idx, return_inverse=True)
        rows = np.asarray(atlas.obsm['norm_msmu'][unique_idx], dtype=np.float32)[inverse]
        restored = np.empty_like(rows)
        restored[order] = rows
        return restored

    def _atlas_subset(time_value, max_cells, rng_seed, label):
        mask_idx = np.flatnonzero(np.isclose(time_values, float(time_value)))
        if mask_idx.size == 0:
            raise ValueError(f'No cells found for time={time_value} in {label}.')
        picked_idx = sample_index_array(mask_idx, max_cells=max_cells, seed=rng_seed)
        msmu = _load_msmu_rows(picked_idx)
        var_copy = atlas_var.copy()
        var_copy.index = pd.Index(gene_names, dtype=str)
        var_copy.index.name = None
        obs = atlas.obs.iloc[picked_idx].copy().reset_index(drop=True)
        sub = _ad.AnnData(X=collapse_msmu(msmu), obs=obs, var=var_copy)
        sub.obsm['norm_msmu'] = msmu
        return sub

    prev_train, next_train = neighboring_train_times(focus_time, train_times)
    start_sub = _atlas_subset(prev_train, max_cells_per_group, seed, 'atlas start stage')
    end_sub = _atlas_subset(next_train, max_cells_per_group, seed + 1, 'atlas end stage')
    gt_sub = _atlas_subset(focus_time, max_cells_per_group, seed + 2, 'atlas ground truth')
    navigo_sub = subset_for_time(pred_adata, focus_time, max_cells_per_group, seed, 'Navigo prediction')

    panel_expr, panel_obs_list = [], []

    def _add_block(adata_block, role, method):
        expr, _ = extract_gene_expression(adata_block, reference_genes=gene_names)
        obs = adata_block.obs.copy().reset_index(drop=True)
        obs['role'] = role
        obs['method'] = method
        if 'day' not in obs.columns:
            obs['day'] = model_to_day.get(float(focus_time), str(focus_time))
        panel_expr.append(expr.astype(np.float32))
        panel_obs_list.append(obs)

    _add_block(start_sub, 'observed_start', 'observed')
    _add_block(end_sub, 'observed_end', 'observed')
    _add_block(gt_sub, 'ground_truth', 'ground_truth')
    _add_block(navigo_sub, 'prediction', 'Navigo')

    if optional_methods:
        for idx, (method_name, method_adata) in enumerate(optional_methods.items()):
            method_sub = subset_for_time(method_adata, focus_time, max_cells_per_group, seed + 10 + idx, method_name)
            _add_block(method_sub, 'prediction', method_name)

    umap_obs = pd.concat(panel_obs_list, ignore_index=True)
    umap_adata = _ad.AnnData(X=np.vstack(panel_expr), obs=umap_obs, var=pd.DataFrame(index=gene_names))
    sc.pp.pca(umap_adata, n_comps=min(50, umap_adata.n_obs - 1, umap_adata.n_vars))
    sc.pp.neighbors(
        umap_adata,
        n_neighbors=min(30, max(5, umap_adata.n_obs - 1)),
        n_pcs=min(30, umap_adata.obsm['X_pca'].shape[1]),
        random_state=seed,
    )
    sc.tl.umap(umap_adata, random_state=seed, min_dist=0.5)
    umap_adata.uns['focus_time'] = float(focus_time)
    umap_adata.uns['model_to_day'] = model_to_day
    umap_adata.uns['prev_train'] = float(prev_train)
    umap_adata.uns['next_train'] = float(next_train)
    return umap_adata


# ---------------------------------------------------------------------------
# Denoising / imputation
# ---------------------------------------------------------------------------

def denoise_trajectory(
    data_path,
    ckpt_path,
    source_indices,
    pred_dir,
    knn_max_cells=80000,
    n_neighbors=20,
    flow_steps=10,
    skip_existing=True,
    seed=42,
):
    """Run Navigo ODE denoising inference along a temporal trajectory.

    Loads the full atlas, trains a KNN classifier on cell-type labels, then for
    each source interval ``(i, i+1)`` in ``source_indices``, integrates the ODE
    from time *i* to time *i+1* and assigns predicted cell types via KNN.
    Saves one ``h5ad`` per interval to ``pred_dir``.

    Parameters
    ----------
    data_path : str or Path
        Path to the full ``.h5ad`` atlas (with ``Ms`` / ``Mu`` layers).
    ckpt_path : str or Path
        Path to the trained checkpoint (``.pth``).
    source_indices : list of int
        Source time indices; interval is ``t_i → t_{i+1}``.
    pred_dir : str or Path
        Output directory for prediction ``.h5ad`` files.
    knn_max_cells : int
        Subsample the KNN training set to at most this many cells.
    n_neighbors : int
        KNN *k*.
    flow_steps : int
        ODE integration steps passed to ``Navigo``.
    skip_existing : bool
        Skip intervals whose output file already exists.
    seed : int

    Returns
    -------
    dict
        ``{'prediction_files_available': int, 'prediction_files_expected': int}``

    Examples
    --------
    >>> summary = navigo.tl.denoise_trajectory(
    ...     FULL_DATA, CKPT, source_indices=SOURCE_INDICES, pred_dir=PRED_DIR,
    ... )
    """
    import torch
    import anndata as _ad
    from pathlib import Path
    from sklearn.neighbors import KNeighborsClassifier
    from navigo.model import MLPTimeGRN, Navigo
    from navigo.io import load_and_preprocess_data

    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [pred_dir / f'pred_t{i}_to_t{i+1}.h5ad' for i in source_indices]
    existing_count = sum(int(p.exists()) for p in expected_files)
    if skip_existing and existing_count == len(expected_files):
        return {'prediction_files_available': existing_count, 'prediction_files_expected': len(expected_files)}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPTimeGRN.load_from_checkpoint(ckpt_path, device)
    model.eval()
    flow = Navigo(model=model, num_steps=flow_steps, device=device)

    data, time_label, adata, _ = load_and_preprocess_data(data_path)
    X_train_full = (data[:, :adata.n_vars] + data[:, adata.n_vars:]).cpu().numpy()
    y_train = adata.obs['cell_type'].values

    if knn_max_cells and len(y_train) > knn_max_cells:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(y_train), size=knn_max_cells, replace=False)
        X_train, y_train_fit = X_train_full[keep_idx], y_train[keep_idx]
    else:
        X_train, y_train_fit = X_train_full, y_train

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train_fit)

    all_time_set = set(float(x) for x in np.sort(np.unique(time_label.numpy())).tolist())

    for i in source_indices:
        source_time, target_time = float(i), float(i + 1)
        out_file = pred_dir / f'pred_t{i}_to_t{i+1}.h5ad'
        if skip_existing and out_file.exists():
            continue
        if source_time not in all_time_set or target_time not in all_time_set:
            continue

        source_mask = time_label == source_time
        z0 = data[source_mask].to(device)
        t0 = torch.tensor([source_time] * z0.shape[0], device=device)
        t1 = torch.tensor([target_time] * z0.shape[0], device=device)

        pred_forward = flow.sample_ode_time_interval(z_full=z0, t_start=t0, t_end=t1, N=100)
        pred_expr = pred_forward[:, :adata.n_vars] + pred_forward[:, adata.n_vars:]
        y_pred = knn.predict(pred_expr)

        pred_out = _ad.AnnData(X=pred_expr)
        pred_out.obs['predicted_cell_type'] = y_pred
        pred_out.obs['source_time'] = source_time
        pred_out.obs['target_time'] = target_time
        pred_out.write_h5ad(out_file)

    return {
        'prediction_files_available': sum(int(p.exists()) for p in expected_files),
        'prediction_files_expected': len(expected_files),
    }


def compute_deg_by_day(
    adata_ct,
    deg_dir,
    gene_names=None,
    file_prefix='',
    day_col='day',
    min_cells=3,
    skip_existing=True,
):
    """Compute per-day Wilcoxon DEGs (target day vs all other days) and save CSVs.

    Parameters
    ----------
    adata_ct : anndata.AnnData
        Cell-type-filtered AnnData with normalised expression in ``X``.
    deg_dir : str or Path
        Output directory for CSV files.
    gene_names : array-like of str, optional
        If ``None``, uses ``adata_ct.var_names``.
    file_prefix : str
        Prefix prepended to each output filename.
    day_col : str
        Column in ``obs`` with day labels.
    min_cells : int
        Skip days with fewer than this many cells.
    skip_existing : bool

    Returns
    -------
    list of Path
        Saved (or pre-existing) DEG CSV file paths.

    Examples
    --------
    >>> saved = navigo.tl.compute_deg_by_day(adata_ct, DEG_DIR, file_prefix='Myofibroblasts_')
    """
    import re
    from pathlib import Path
    from navigo.io import to_dense
    from navigo.metrics import wilcoxon_deg

    deg_dir = Path(deg_dir)
    deg_dir.mkdir(parents=True, exist_ok=True)

    if gene_names is None:
        gene_names = np.asarray(adata_ct.var_names.astype(str))

    x_ct = to_dense(adata_ct.X)
    day_values = adata_ct.obs[day_col].astype(str).values

    def _day_sort_key(day):
        day = str(day)
        if day == 'P0':
            return 19.5
        m = re.search(r'-?\d+(?:\.\d+)?', day)
        return float(m.group()) if m else 0.0

    days = sorted(set(day_values), key=_day_sort_key)
    saved = []

    for day in days:
        fname = f'{file_prefix}{day.replace("P0", "E19.5")}_deg.csv'
        out_file = deg_dir / fname
        if skip_existing and out_file.exists():
            saved.append(out_file)
            continue
        day_mask = day_values == day
        other_mask = ~day_mask
        if int(day_mask.sum()) < min_cells or int(other_mask.sum()) < min_cells:
            continue
        df_deg = wilcoxon_deg(x_ct[day_mask], x_ct[other_mask], gene_names)
        df_deg.to_csv(out_file, index=False)
        saved.append(out_file)

    return saved


# ---------------------------------------------------------------------------
# GRN analysis
# ---------------------------------------------------------------------------

def compute_grn_expression_changes(
    adata_subset,
    ckpt_path,
    raw_out_dir,
    result_table_dir,
    cell_type_tag,
    ko_multiplier=0.0,
    device=None,
    flow_steps=100,
    simulation_steps=10,
):
    """Enrich raw knockout CSVs with wild-type expression and relative change.

    Loads the checkpoint, runs one step of wild-type ODE integration to obtain
    per-gene mean expression, then annotates every raw knockout CSV in
    ``raw_out_dir`` with ``wt_total_expr`` and ``relative_change`` columns and
    saves the enriched files to ``result_table_dir``.

    Parameters
    ----------
    adata_subset : anndata.AnnData
        Cell-type-filtered AnnData with ``layers['Ms']`` and ``layers['Mu']``.
    ckpt_path : str or Path
        Checkpoint path.
    raw_out_dir : str or Path
        Directory containing raw knockout CSVs from :func:`navigo.run_perturbation_inference`.
    result_table_dir : str or Path
        Output directory for enriched CSVs.
    cell_type_tag : str
        Tag used in output filenames (e.g. ``'First_heart_field'``).
    ko_multiplier : float
        Multiplier string for filename (e.g. ``0.0`` → ``'neg0.0x'``).
    device : str or None
        ``'cuda'`` or ``'cpu'``. Auto-detected if ``None``.
    flow_steps : int
    simulation_steps : int

    Returns
    -------
    list of Path
        Saved enriched CSV paths.

    Examples
    --------
    >>> saved = navigo.tl.compute_grn_expression_changes(
    ...     adata_subset, CHECKPOINT_PATH, RAW_OUT, RESULT_TABLE_DIR, CELL_TYPE_TAG,
    ... )
    """
    import torch
    from pathlib import Path
    from navigo.model import MLPTimeGRN, Navigo

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_out_dir = Path(raw_out_dir)
    result_table_dir = Path(result_table_dir)
    result_table_dir.mkdir(parents=True, exist_ok=True)

    input_dim = int(adata_subset.n_vars * 2)
    gene_names = list(
        adata_subset.var['gene_name'].astype(str).values
        if 'gene_name' in adata_subset.var.columns
        else adata_subset.var_names.astype(str)
    )

    ms = np.asarray(adata_subset.layers['Ms'])
    mu = np.asarray(adata_subset.layers['Mu'])
    data_np = np.concatenate([ms, mu], axis=1)
    data_np = (data_np - data_np.min(axis=0)) / (data_np.max(axis=0) - data_np.min(axis=0) + 1e-7)

    model = MLPTimeGRN.load_from_checkpoint(ckpt_path, device)
    model.eval()
    flow = Navigo(model=model, num_steps=flow_steps, device=device)

    data_t = torch.tensor(data_np, dtype=torch.float32).to(device)
    time_t = torch.tensor(np.asarray(adata_subset.obs['time'], dtype=np.float32), dtype=torch.float32).to(device)

    pred_base = flow.sample_ode_time_interval(
        z_full=data_t, t_start=time_t, t_end=time_t + 1, N=simulation_steps,
    )

    half = input_dim // 2
    wt_vals = pred_base.mean(axis=0)[:half] + pred_base.mean(axis=0)[half:]
    wt_map = {g: float(v) for g, v in zip(gene_names, wt_vals)}

    mult_str = f'neg{abs(ko_multiplier)}x'
    saved = []
    for csv_path in sorted(raw_out_dir.glob('*.csv')):
        if csv_path.name == 'inference_summary.json':
            continue
        target = csv_path.stem
        df = pd.read_csv(csv_path)
        if 'gene_name' not in df.columns and 'gene' in df.columns:
            df = df.rename(columns={'gene': 'gene_name'})

        df['wt_total_expr'] = df['gene_name'].map(wt_map).fillna(0.0).astype(float)
        df.loc[df['gene_name'] == target, 'wt_total_expr'] = 0.0
        df['relative_change'] = np.where(
            df['wt_total_expr'] != 0,
            df['total_change'] / (df['wt_total_expr'] + 1e-7),
            0.0,
        )
        df = df.sort_values('total_change', key=lambda s: np.abs(s), ascending=False)
        out_df = df[['gene_name', 'spliced_change', 'unspliced_change', 'total_change', 'relative_change', 'wt_total_expr']].rename(columns={'gene_name': 'gene'})
        out_name = f'{target}_{mult_str}_knockout_{cell_type_tag}.csv'
        out_df.to_csv(result_table_dir / out_name, index=False)
        saved.append(result_table_dir / out_name)

    return saved


def run_grn_analysis_scripts(
    summary_ws,
    result_table_dir,
    grn_script_sources,
    asset_sources=None,
    extra_panel_barplot_files=None,
    expected_outputs=None,
):
    """Orchestrate the GRN CHD analysis helper scripts.

    Copies results and scripts to a structured workspace, patches them for
    anndata compatibility and a deterministic shared UMAP layout, then runs
    each script via subprocess.

    Parameters
    ----------
    summary_ws : str or Path
        Root workspace directory (will be created if needed).
    result_table_dir : str or Path
        Directory containing enriched knockout CSVs from
        :func:`compute_grn_expression_changes`.
    grn_script_sources : dict of {str: Path}
        Maps script filename → source path within the navigo package.
    asset_sources : dict of {str: Path}, optional
        Extra data files to copy into the workspace root.
    extra_panel_barplot_files : dict of {str: Path}, optional
        Extra files to copy into ``Panel_barplot/``.
    expected_outputs : list of Path, optional
        If all exist, skip re-running. Defaults to the six main figure files.

    Returns
    -------
    dict with keys ``'skipped'`` (bool) and ``'workspace'`` (Path).

    Examples
    --------
    >>> result = navigo.tl.run_grn_analysis_scripts(
    ...     SUMMARY_WS, RESULT_TABLE_DIR, grn_script_sources,
    ...     asset_sources=asset_sources,
    ... )
    """
    import os
    import re
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    summary_ws = Path(summary_ws)
    result_table_dir = Path(result_table_dir)

    if expected_outputs is None:
        expected_outputs = [
            summary_ws / 'Panel_UMAP' / 'figures' / 'umapumap_clusters.png',
            summary_ws / 'Panel_UMAP' / 'interaction_network_v3.pdf',
            summary_ws / 'Panel_UMAP' / 'cluster_results' / 'chd_cluster_heatmap_final.png',
        ]

    if all(Path(p).exists() for p in expected_outputs):
        return {'skipped': True, 'workspace': summary_ws}

    for sub in ['Panel_UMAP', 'Panel_pathway_enrichment', 'Panel_barplot', 'results']:
        (summary_ws / sub).mkdir(parents=True, exist_ok=True)
    (summary_ws / 'Panel_pathway_enrichment' / 'pathway_enrichment').mkdir(parents=True, exist_ok=True)

    for f in result_table_dir.glob('*_neg0.0x_knockout_First_heart_field.csv'):
        shutil.copy2(f, summary_ws / 'results' / f.name)

    if asset_sources:
        for dest_name, src_path in asset_sources.items():
            shutil.copy2(src_path, summary_ws / dest_name)

    umap_scripts = ['run_umap.py', 'plot_umap.py', 'analyze_chd_distance.py',
                    'plot_chd_distance.py', 'analyze_chd_distribution.py', 'plot_umap_interaction_v3.py']
    for script_name in umap_scripts:
        if script_name in grn_script_sources:
            shutil.copy2(grn_script_sources[script_name], summary_ws / 'Panel_UMAP' / script_name)

    if 'pathway_enrichment.py' in grn_script_sources:
        shutil.copy2(grn_script_sources['pathway_enrichment.py'],
                     summary_ws / 'Panel_pathway_enrichment' / 'pathway_enrichment.py')

    for script_name in ['analyze_chd_clean.py', 'analyze_chd_clean_dynamo.py']:
        if script_name in grn_script_sources:
            shutil.copy2(grn_script_sources[script_name], summary_ws / 'Panel_barplot' / script_name)

    if extra_panel_barplot_files:
        for dest_name, src_path in extra_panel_barplot_files.items():
            shutil.copy2(src_path, summary_ws / 'Panel_barplot' / dest_name)

    def _patch_anndata_compat(path):
        txt = Path(path).read_text()
        if 'import scanpy as sc' not in txt:
            return
        shim = (
            "import anndata as _anndata_compat\n"
            "if not hasattr(_anndata_compat, 'read'):\n"
            "    _anndata_compat.read = _anndata_compat.read_h5ad\n"
        )
        if "_anndata_compat.read = _anndata_compat.read_h5ad" not in txt:
            txt = txt.replace('import scanpy as sc', shim + 'import scanpy as sc')
        Path(path).write_text(txt)

    def _patch_deterministic_umap(plot_umap_path, interaction_path, run_umap_path):
        for p in [plot_umap_path, interaction_path, run_umap_path]:
            txt = Path(p).read_text()
            txt = txt.replace(
                'ko_files = glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv")',
                'ko_files = sorted(glob(f"{ko_path}/*_neg0.0x_knockout_First_heart_field.csv"))',
            )
            Path(p).write_text(txt)

        txt = Path(plot_umap_path).read_text()
        if 'umap_embedding.csv' not in txt:
            marker = "    adata = anndata.AnnData(X=response_matrix)"
            insert = (
                "    emb_df = pd.DataFrame({'gene': gene_names, 'cluster': clusters,"
                " 'umap1': response_umap[:, 0], 'umap2': response_umap[:, 1]})\n"
                "    emb_df.to_csv('umap_embedding.csv', index=False)\n\n"
            )
            txt = txt.replace(marker, insert + marker)
            Path(plot_umap_path).write_text(txt)

        txt = Path(interaction_path).read_text()
        if "embed_path = 'umap_embedding.csv'" not in txt:
            old = (
                "    reducer = umap.UMAP(n_components=2, random_state=42)\n"
                "    response_umap = reducer.fit_transform(response_matrix)\n"
            )
            new = (
                "    embed_path = 'umap_embedding.csv'\n"
                "    if os.path.exists(embed_path):\n"
                "        emb = pd.read_csv(embed_path).set_index('gene')\n"
                "        if set(gene_names).issubset(set(emb.index)):\n"
                "            emb = emb.loc[gene_names]\n"
                "            response_umap = emb[['umap1', 'umap2']].values\n"
                "            clusters = emb['cluster'].astype(int).values\n"
                "        else:\n"
                "            reducer = umap.UMAP(n_components=2, random_state=42)\n"
                "            response_umap = reducer.fit_transform(response_matrix)\n"
                "    else:\n"
                "        reducer = umap.UMAP(n_components=2, random_state=42)\n"
                "        response_umap = reducer.fit_transform(response_matrix)\n"
            )
            txt = txt.replace(old, new)
            Path(interaction_path).write_text(txt)

    for sp in [summary_ws / 'Panel_UMAP' / 'plot_umap.py',
               summary_ws / 'Panel_UMAP' / 'plot_umap_interaction_v3.py']:
        if sp.exists():
            _patch_anndata_compat(sp)

    _patch_deterministic_umap(
        summary_ws / 'Panel_UMAP' / 'plot_umap.py',
        summary_ws / 'Panel_UMAP' / 'plot_umap_interaction_v3.py',
        summary_ws / 'Panel_UMAP' / 'run_umap.py',
    )

    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'

    def _run(script, cwd):
        proc = subprocess.run([sys.executable, script], cwd=cwd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            if proc.stdout.strip():
                print(proc.stdout[-1000:])
            print(proc.stderr)
            raise RuntimeError(f'Failed: {script} in {cwd}')

    _run('run_umap.py', summary_ws / 'Panel_UMAP')
    _run('plot_umap.py', summary_ws / 'Panel_UMAP')
    _run('analyze_chd_distance.py', summary_ws / 'Panel_UMAP')
    _run('plot_chd_distance.py', summary_ws / 'Panel_UMAP')
    _run('analyze_chd_distribution.py', summary_ws / 'Panel_UMAP')
    _run('plot_umap_interaction_v3.py', summary_ws / 'Panel_UMAP')
    _run('pathway_enrichment.py', summary_ws / 'Panel_pathway_enrichment')
    _run('analyze_chd_clean.py', summary_ws / 'Panel_barplot')
    _run('analyze_chd_clean_dynamo.py', summary_ws / 'Panel_barplot')

    return {'skipped': False, 'workspace': summary_ws}


def evaluate_reprogramming_screen(
    data_path,
    inference_dir,
    metrics_path,
    groundtruth_path,
    output_dir,
    thresholds=None,
    min_expr_values=None,
    exclude_genes=None,
):
    """Evaluate a virtual reprogramming screen via ROC analysis.

    Computes overall AUROC and per-TF/target AUROCs across all combinations of
    programming efficiency thresholds and expression filters. Saves per-setting
    ROC curve CSVs, per-TF/target AUROC tables, and summary pivot tables.

    Parameters
    ----------
    data_path : str or Path
        Path to the fibroblast h5ad (with Ms/Mu layers).
    inference_dir : str or Path
        Directory with per-target knockout CSVs.
    metrics_path : str or Path
        CSV of in-silico accuracy scores (from perturbation inference).
    groundtruth_path : str or Path
        CSV of experimental ground-truth hit labels.
    output_dir : str or Path
        Output directory for result tables.
    thresholds : list of float, optional
        Programming efficiency thresholds. Default: [0, 0.2, 0.4, 0.6, 0.8, 1.0].
    min_expr_values : list of float, optional
        Minimum expression filters. Default: [0, 0.005, 0.01, 0.015, 0.02].
    exclude_genes : list of str, optional
        Genes to exclude from the evaluation matrix.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (threshold, min_expr) combination with AUROC and correlation metrics.

    Examples
    --------
    >>> summary_df = navigo.tl.evaluate_reprogramming_screen(
    ...     DATA_PATH, RERUN_DIR, METRICS_PATH, GROUNDTRUTH_PATH, OUTPUT_DIR,
    ... )
    """
    import anndata as _ad
    from pathlib import Path
    from sklearn.metrics import roc_curve, auc as sk_auc
    from scipy.stats import spearmanr

    if thresholds is None:
        thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if min_expr_values is None:
        min_expr_values = [0, 0.005, 0.01, 0.015, 0.02]
    if exclude_genes is None:
        exclude_genes = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = _ad.read_h5ad(data_path)
    adata = adata[adata.obs['cell_type'].astype(str) == 'Fibroblasts'].copy()
    joint = np.concatenate([adata.layers['Ms'], adata.layers['Mu']], axis=1)
    joint = (joint - joint.min(axis=0)) / (joint.max(axis=0) - joint.min(axis=0) + 1e-7)
    num_genes = joint.shape[1] // 2
    expression = (joint[:, :num_genes] + joint[:, num_genes:]).mean(axis=0)
    gene_names_arr = (
        adata.var['gene_name'].astype(str).to_numpy()
        if 'gene_name' in adata.var.columns
        else adata.var_names.astype(str).to_numpy()
    )

    df_insilico = pd.read_csv(metrics_path, index_col=0)
    df_real = pd.read_csv(groundtruth_path, index_col=0)

    genes = set(df_real.index) | set(df_real.columns)
    gene_name_set = set(gene_names_arr)
    gene_expr = {
        g: float(expression[np.where(gene_names_arr == g)[0][0]])
        for g in genes if g in gene_name_set
    }
    expression_df = pd.DataFrame.from_dict(gene_expr, orient='index', columns=['expression'])

    df_split = df_insilico.index.to_series().str.split('_', expand=True)
    df_insilico[['tf1', 'tf2']] = df_split
    heatmap_full = df_insilico.pivot(index='tf1', columns='tf2', values='avg_accuracy').reindex(
        index=df_real.index, columns=df_real.columns
    )

    if exclude_genes:
        heatmap_full = heatmap_full[~heatmap_full.index.isin(exclude_genes)]
        df_real = df_real[~df_real.index.isin(exclude_genes)]

    def _calc_auc(y_true, y_score):
        if len(np.unique(y_true)) <= 1:
            return None
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(sk_auc(fpr, tpr))

    rows = []
    for threshold in thresholds:
        for min_expr in min_expr_values:
            def _expr_ok(g):
                return g in expression_df.index and expression_df.loc[g, 'expression'] >= min_expr

            hm = heatmap_full.loc[
                [g for g in heatmap_full.index if _expr_ok(g)],
                [g for g in heatmap_full.columns if _expr_ok(g)],
            ]
            gt = df_real.loc[
                [g for g in df_real.index if _expr_ok(g)],
                [g for g in df_real.columns if _expr_ok(g)],
            ]

            y_true = (gt > threshold).values.flatten()
            y_score = hm.values.flatten()
            fpr, tpr, _ = roc_curve(y_true, y_score)
            overall_auc = float(sk_auc(fpr, tpr))

            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(
                output_dir / f'02_roc_curve_thr_{threshold}_minexpr_{min_expr}.csv', index=False
            )

            tf_rows, target_rows = [], []
            for gene in gt.index:
                a = _calc_auc((gt.loc[gene] > threshold).astype(int), hm.loc[gene])
                if a is not None:
                    tf_rows.append({'gene': gene, 'auroc': a,
                                    'expression': float(expression_df.loc[gene, 'expression'])})
            for gene in gt.columns:
                a = _calc_auc((gt[gene] > threshold).astype(int), hm[gene])
                if a is not None:
                    target_rows.append({'gene': gene, 'auroc': a,
                                        'expression': float(expression_df.loc[gene, 'expression'])})

            tf_df = pd.DataFrame(tf_rows)
            target_df = pd.DataFrame(target_rows)
            tf_df.to_csv(output_dir / f'02_tf_auroc_expression_thr_{threshold}_minexpr_{min_expr}.csv', index=False)
            target_df.to_csv(output_dir / f'02_target_auroc_expression_thr_{threshold}_minexpr_{min_expr}.csv', index=False)

            tf_corr, tf_p = spearmanr(tf_df['auroc'], tf_df['expression']) if len(tf_df) > 1 else (np.nan, np.nan)
            target_corr, target_p = spearmanr(target_df['auroc'], target_df['expression']) if len(target_df) > 1 else (np.nan, np.nan)

            rows.append({
                'threshold': threshold, 'min_expr': min_expr, 'overall_auc': overall_auc,
                'tf_corr': float(tf_corr) if not np.isnan(tf_corr) else np.nan,
                'tf_pval': float(tf_p) if not np.isnan(tf_p) else np.nan,
                'tf_count': int(len(tf_df)),
                'target_corr': float(target_corr) if not np.isnan(target_corr) else np.nan,
                'target_pval': float(target_p) if not np.isnan(target_p) else np.nan,
                'target_count': int(len(target_df)),
                'matrix_rows': int(hm.shape[0]), 'matrix_cols': int(hm.shape[1]),
            })

    summary_df = pd.DataFrame(rows)

    # Build in-memory pivot tables for direct use
    auroc_heatmap = summary_df.pivot(index='threshold', columns='min_expr', values='overall_auc')

    # Store ROC curves keyed by (threshold, min_expr)
    roc_curves = {}
    for threshold in thresholds:
        for min_expr in min_expr_values:
            csv_path = output_dir / f'02_roc_curve_thr_{threshold}_minexpr_{min_expr}.csv'
            if csv_path.exists():
                roc_curves[(threshold, min_expr)] = pd.read_csv(csv_path)

    return {
        'summary': summary_df,
        'auroc_heatmap': auroc_heatmap,
        'roc_curves': roc_curves,
    }


# ---------------------------------------------------------------------------
# Denoising analysis — individual steps (return in-memory data)
# ---------------------------------------------------------------------------

def _run_script_via_argv(main_func, argv_list):
    """Call an argparse-based main() by temporarily replacing sys.argv."""
    import sys
    saved = sys.argv
    sys.argv = [''] + argv_list
    try:
        main_func()
    finally:
        sys.argv = saved


def extract_denoising_markers(
    full_data, pred_dir, deg_dir, cell_type, target_day, start_day, step,
    cache_dir=None,
):
    """Extract marker genes at a target timepoint.

    Compares predicted and observed expression across a window of 4
    neighbouring time points to identify genes whose expression is
    recovered by the denoised prediction.

    Parameters
    ----------
    full_data : str or Path
        Full developmental AnnData (with ``Ms``/``Mu`` layers).
    pred_dir, deg_dir : str or Path
    cell_type : str
    target_day, start_day, step : float
    cache_dir : str or Path, optional
        If given, saves a CSV cache and skips re-computation on the next call.

    Returns
    -------
    pd.DataFrame
        Marker gene expression at 4 time points + prediction.
    """
    from pathlib import Path
    import tempfile

    ct_file = cell_type.replace('/', '_').replace(' ', '_').replace('(', '|').replace(')', '|')
    point = int(round((target_day - start_day) / step))

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cache_dir / f'{ct_file}_marker_genes_t{point}.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)

    out_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
    csv_path = out_dir / f'{ct_file}_marker_genes_t{point}.csv'

    from navigo.interpolation_case_panel_marker_gene import main as _main
    _run_script_via_argv(_main, [
        '--input_data', str(full_data), '--pred_dir', str(pred_dir),
        '--deg_dir', str(deg_dir), '--cell_type', cell_type,
        '--target_day', str(target_day), '--start_day', str(start_day),
        '--step', str(step), '--output_csv', str(csv_path),
    ])
    return pd.read_csv(csv_path)


def denoising_pathway_enrichment(
    full_data, pred_dir, deg_dir, msigdb_path, cell_type, target_day,
    start_day, step, cache_dir=None,
):
    """Compute pathway enrichment for predicted vs observed cells at each day.

    Uses Mann-Whitney rank-sum + Fisher's exact test to identify GOBP
    pathways enriched in the predicted cell population.

    Parameters
    ----------
    full_data, pred_dir, deg_dir, msigdb_path : str or Path
    cell_type : str
    target_day, start_day, step : float
    cache_dir : str or Path, optional

    Returns
    -------
    pd.DataFrame
        Shared pathways with enrichment scores across time points.
    """
    from pathlib import Path
    import tempfile

    ct_file = cell_type.replace('/', '_').replace(' ', '_').replace('(', '|').replace(')', '|')

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        shared_csv = cache_dir / f'{ct_file}_shared_pathways.csv'
        if shared_csv.exists():
            return pd.read_csv(shared_csv)

    out_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
    shared_csv = out_dir / f'{ct_file}_shared_pathways.csv'

    from navigo.interpolation_case_panel_pathway_enrichment import main as _main
    _run_script_via_argv(_main, [
        '--input_data', str(full_data), '--pred_dir', str(pred_dir),
        '--deg_dir', str(deg_dir), '--msigdb_path', str(msigdb_path),
        '--cell_type', cell_type, '--target_day', str(target_day),
        '--start_day', str(start_day), '--step', str(step),
        '--output_dir', str(out_dir),
    ])
    return pd.read_csv(shared_csv)


def build_denoising_marker_table(
    full_data, pred_dir, cell_type, target_day, start_day, step,
):
    """Build marker expression display table from raw data and predictions.

    Computes mean expression per marker gene at 4 time points (observed)
    plus the denoised prediction, returning two scaled DataFrames ready
    for heatmap plotting.

    Parameters
    ----------
    full_data, pred_dir : str or Path
    cell_type : str
    target_day, start_day, step : float

    Returns
    -------
    dict with keys:
        ``'display_df'`` — raw expression DataFrame,
        ``'noisy_scaled'`` — 0–1 scaled observed expression (DataFrame),
        ``'navigo_scaled'`` — 0–1 scaled predicted expression (DataFrame),
        ``'genes'`` — list of gene names used.
    """
    from pathlib import Path
    import tempfile
    from navigo.interpolation_case_render_end_to_end_figures import build_marker_display_tables

    with tempfile.TemporaryDirectory() as tmpdir:
        result = build_marker_display_tables(
            input_data=Path(full_data), pred_dir=Path(pred_dir),
            cell_type=cell_type, target_day=target_day,
            start_day=start_day, step=step, table_dir=Path(tmpdir),
        )
        display_df = pd.read_csv(result['display_table'])

    return {
        'display_df': display_df,
        'noisy_scaled': result['noisy_scaled'],
        'navigo_scaled': result['navigo_scaled'],
        'genes': result['available_genes'],
    }


def build_denoising_pathway_table(shared_pathways_df):
    """Build pathway enrichment plot table from shared pathways data.

    Filters to the panel-focus pathways and computes −log₁₀(p) columns
    for Navigo (predicted) vs Noisy (observed at target day).

    Parameters
    ----------
    shared_pathways_df : pd.DataFrame
        Output of :func:`denoising_pathway_enrichment`.

    Returns
    -------
    pd.DataFrame
        With columns ``pathway``, ``pathway_short``, ``Navigo``, ``Noisy``.
    """
    from navigo.interpolation_case_render_end_to_end_figures import (
        PANEL_FOCUS, _abbreviate_pathway,
    )

    plot_df = shared_pathways_df[shared_pathways_df['pathway'].isin(PANEL_FOCUS)].copy()
    plot_df = plot_df.set_index('pathway').reindex(PANEL_FOCUS).reset_index()
    plot_df['pathway_short'] = plot_df['pathway'].map(_abbreviate_pathway)
    plot_df['Navigo'] = plot_df['pred_E18.25_neg_log_pval'].fillna(0.0)
    plot_df['Noisy'] = plot_df['real_E18.25_neg_log_pval'].fillna(0.0)
    return plot_df[['pathway', 'pathway_short', 'Navigo', 'Noisy']]


def build_denoising_trajectory_table(
    full_data, pred_dir, ct_to_traj_json, cell_type,
    day_min=14.0, day_max=19.5,
):
    """Build trajectory proportion table from raw data and predictions.

    Counts cell-type composition for observed ("Noisy") and predicted
    ("Navigo") cells at each developmental day.

    Parameters
    ----------
    full_data, pred_dir, ct_to_traj_json : str or Path
    cell_type : str
    day_min, day_max : float

    Returns
    -------
    pd.DataFrame
        With columns ``series``, ``day_num``, ``day_label``, ``group``,
        ``count``, ``ratio``, ``source``.
    """
    from pathlib import Path
    import tempfile
    from navigo.interpolation_case_render_end_to_end_figures import build_trajectory_plot_table

    with tempfile.TemporaryDirectory() as tmpdir:
        _, trajectory_df = build_trajectory_plot_table(
            input_data=Path(full_data), pred_dir=Path(pred_dir),
            ct_to_trajectory_json=Path(ct_to_traj_json),
            cell_type=cell_type, day_min=day_min, day_max=day_max,
            table_dir=Path(tmpdir),
        )
    return trajectory_df


# ---------------------------------------------------------------------------
# Training demo helpers
# ---------------------------------------------------------------------------

def parse_training_log(log_path):
    """Parse training loss records from a Navigo training log file.

    Parameters
    ----------
    log_path : str or Path
        Path to the stdout log produced by ``submission/main_navigo.py``.

    Returns
    -------
    pd.DataFrame
        One row per logged step with columns ``train_step``, ``all_loss``, etc.
    """
    import re
    from pathlib import Path

    log_path = Path(log_path)
    if not log_path.exists():
        return pd.DataFrame()

    records = []
    text = log_path.read_text(encoding='utf-8', errors='ignore').replace('\r', '\n')
    for line in text.splitlines():
        if 'all_loss:' not in line:
            continue
        pairs = re.findall(r'([A-Za-z0-9_]+):\s*([0-9.]+)', line)
        if not pairs:
            continue
        row = {key: float(value) for key, value in pairs}
        row['train_step'] = len(records) + 1
        records.append(row)
    return pd.DataFrame(records)


def summarize_round_scores(score_dir):
    """Summarize per-round EMD scores from a training output directory.

    Parameters
    ----------
    score_dir : str or Path
        Directory containing ``score_*.json`` files.

    Returns
    -------
    round_df : pd.DataFrame
        One row per round with mean EMD metrics.
    final_detail_df : pd.DataFrame
        Per-time detail from the final round.
    """
    import json
    from pathlib import Path

    score_dir = Path(score_dir)
    round_rows = []
    final_detail_df = pd.DataFrame()

    for score_path in sorted(score_dir.glob('score_*.json')):
        round_idx = int(score_path.stem.split('_')[-1]) + 1
        payload = json.loads(score_path.read_text())
        detail = []
        for t_str, m in sorted(payload.items(), key=lambda x: float(x[0])):
            baseline = float(m['baseline'])
            prediction = float(m['prediction'])
            detail.append({
                'round': round_idx, 'time': float(t_str),
                'prediction_emd': prediction, 'baseline_emd': baseline,
                'improvement': baseline - prediction,
                'relative_improvement': (baseline - prediction) / baseline if baseline > 0 else 0.0,
            })
        detail_df = pd.DataFrame(detail).sort_values('time').reset_index(drop=True)
        if detail_df.empty:
            continue
        round_rows.append({
            'round': round_idx,
            'mean_prediction_emd': detail_df['prediction_emd'].mean(),
            'mean_baseline_emd': detail_df['baseline_emd'].mean(),
            'mean_improvement': detail_df['improvement'].mean(),
            'mean_relative_improvement': detail_df['relative_improvement'].mean(),
        })
        final_detail_df = detail_df

    return pd.DataFrame(round_rows), final_detail_df


def sample_training_subset(
    input_data, output_data, total_cells=10000, num_timepoints=10, seed=42,
):
    """Sample a training subset from the full atlas.

    Parameters
    ----------
    input_data : str or Path
        Full h5ad atlas.
    output_data : str or Path
        Output path for the sampled subset.
    total_cells : int
    num_timepoints : int
    seed : int

    Returns
    -------
    dict with sampling summary (loaded from the output JSON).
    """
    import json
    from pathlib import Path

    output_data = Path(output_data)
    summary_path = output_data.with_suffix('.json')
    if output_data.exists() and summary_path.exists():
        return json.loads(summary_path.read_text())

    output_data.parent.mkdir(parents=True, exist_ok=True)
    from navigo.training_demo_sample_training_subset import main as _main
    _run_script_via_argv(_main, [
        '--input-data', str(input_data),
        '--output-data', str(output_data),
        '--total-cells', str(total_cells),
        '--num-timepoints', str(num_timepoints),
        '--seed', str(seed),
        '--overwrite',
    ])
    return json.loads(summary_path.read_text())


def validate_training(
    subset_data, full_data, checkpoint, output_dir,
    hidden_1=5012, hidden_2=5012, flow_steps=10, integration_steps=25,
    max_cells_per_group=300, seed=42, device='auto',
):
    """Run held-out intermediate time point validation.

    Parameters
    ----------
    subset_data, full_data, checkpoint : str or Path
    output_dir : str or Path
    hidden_1, hidden_2, flow_steps, integration_steps : int
    max_cells_per_group : int
    seed : int
    device : str

    Returns
    -------
    dict with keys ``'summary'`` (dict) and ``'metrics'`` (DataFrame).
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    csv_path = output_dir / 'heldout_intermediate_metrics.csv'
    json_path = output_dir / 'heldout_intermediate_metrics_summary.json'

    if csv_path.exists() and json_path.exists():
        return {
            'summary': json.loads(json_path.read_text()),
            'metrics': pd.read_csv(csv_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    from navigo.training_demo_validate_intermediate_metrics import main as _main
    _run_script_via_argv(_main, [
        '--subset-data', str(subset_data),
        '--full-data', str(full_data),
        '--checkpoint', str(checkpoint),
        '--output-dir', str(output_dir),
        '--hidden-1', str(hidden_1), '--hidden-2', str(hidden_2),
        '--flow-steps', str(flow_steps),
        '--integration-steps', str(integration_steps),
        '--max-cells-per-group', str(max_cells_per_group),
        '--seed', str(seed), '--device', device,
    ])
    return {
        'summary': json.loads(json_path.read_text()),
        'metrics': pd.read_csv(csv_path),
    }
