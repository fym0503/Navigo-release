"""I/O and data loading utilities for Navigo."""

import numpy as np
import torch
from pathlib import Path


def to_dense(x) -> np.ndarray:
    """Convert a sparse matrix to a dense NumPy array; pass through if already dense."""
    from scipy import sparse
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def gene_names_from_var(var) -> np.ndarray:
    """Extract gene name strings from an AnnData .var DataFrame.

    Prefers the ``gene_name`` column if present; otherwise uses the index.
    """
    if 'gene_name' in var.columns:
        return var['gene_name'].astype(str).to_numpy()
    return var.index.astype(str).to_numpy()


def read_h5ad(path_like):
    """Read an h5ad file, compatible with both legacy and current anndata APIs."""
    import anndata
    path = Path(path_like)
    if hasattr(anndata, 'read_h5ad'):
        return anndata.read_h5ad(path)
    return anndata.read(path)


def load_and_preprocess_data(input_data_path):
    """Load an h5ad file with Ms/Mu layers and return normalized tensors plus metadata.

    Returns
    -------
    data : torch.Tensor, shape (n_cells, 2 * n_genes)
        Min-max normalized concatenation of Ms and Mu layers.
    time_label : torch.Tensor, shape (n_cells,)
        Cell time labels from ``adata.obs['time']``.
    adata : anndata.AnnData
        The loaded AnnData object (with original layers intact).
    gene_names : np.ndarray, shape (n_genes,)
        Gene name strings extracted from ``adata.var``.

    Notes
    -----
    The returned ``data`` tensor is normalized per-feature (column) using
    min-max scaling with a small epsilon (1e-7) to avoid division by zero.
    This matches the preprocessing in ``submission/main_navigo.py``, and
    extends it by also returning ``adata`` and ``gene_names`` for downstream
    label transfer and DEG analysis.
    """
    import anndata as _ad
    _adata = _ad.read_h5ad(str(input_data_path))

    if 'time' not in _adata.obs:
        raise KeyError("Input AnnData must contain obs['time'].")
    if 'Ms' not in _adata.layers or 'Mu' not in _adata.layers:
        raise KeyError("Input AnnData must contain layers['Ms'] and layers['Mu'].")

    ms = to_dense(_adata.layers['Ms']).astype(np.float32)
    mu = to_dense(_adata.layers['Mu']).astype(np.float32)
    _m = np.concatenate([ms, mu], axis=1)

    _data = torch.tensor(_m, dtype=torch.float32)
    _dmin = _data.amin(dim=0)
    _dmax = _data.amax(dim=0)
    _data = (_data - _dmin) / (_dmax - _dmin).clamp_min(1e-7)

    _tl = torch.tensor(np.asarray(_adata.obs['time'], dtype=np.float32), dtype=torch.float32)
    _gnames = gene_names_from_var(_adata.var)

    return _data, _tl, _adata, _gnames


def normalize_to_x(adata):
    """Normalize Ms/Mu layers and assign their sum to ``adata.X`` in-place.

    Concatenates ``layers['Ms']`` and ``layers['Mu']``, applies min-max
    normalization, then sets ``adata.X = norm_Ms + norm_Mu`` (per-gene sum).
    Modifies ``adata`` in-place and returns it.
    """
    import numpy as _np
    ms = to_dense(adata.layers['Ms']).astype(_np.float32)
    mu = to_dense(adata.layers['Mu']).astype(_np.float32)
    m = _np.concatenate([ms, mu], axis=1)
    denom = m.max(axis=0) - m.min(axis=0) + 1e-7
    norm = (m - m.min(axis=0)) / denom
    n = adata.n_vars
    adata.X = norm[:, :n] + norm[:, n:]
    return adata
