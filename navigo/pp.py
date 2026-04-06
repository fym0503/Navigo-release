"""Preprocessing utilities for Navigo (analogous to sc.pp)."""

import numpy as np

from navigo.io import gene_names_from_var
from navigo.trajectory import prepare_time_axis


def load_atlas(path, backed=True):
    """Load a trajectory atlas h5ad file and prepare its time axis.

    Reads the file, assigns uniform float ``obs['time']`` values from day
    labels (or an existing time column), and stores convenience metadata in
    ``adata.uns``.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5ad`` atlas file.  Should contain either an
        ``obs['day']`` column with stage labels (e.g. ``'E10'``) or a numeric
        ``obs['time']`` column.
    backed : bool
        If ``True`` (default), load the expression matrix in backed mode so
        that only the requested rows are materialised in RAM.  Set to
        ``False`` to load everything into memory.

    Returns
    -------
    adata : anndata.AnnData
        Atlas with ``obs['time']`` (float model time) and ``obs['day']``
        (original stage label) populated.  The following keys are added to
        ``adata.uns``:

        * ``'gene_names'`` – gene name strings (shape ``(n_genes,)``).
        * ``'all_times'``  – sorted unique model time values.
        * ``'model_to_day'`` – dict mapping model time → day label.

    all_times : np.ndarray
        Sorted unique model time values present in the atlas.
    model_to_day : dict
        Maps every model time float to its human-readable day label.

    Examples
    --------
    >>> atlas, all_times, model_to_day = navigo.pp.load_atlas(INPUT_DATA)
    >>> train_times = all_times[::2]
    >>> test_times  = navigo.tl.interior_test_times(all_times, train_times)
    """
    import anndata as _ad
    from pathlib import Path

    path = Path(path)
    adata = _ad.read_h5ad(path, backed='r' if backed else None)

    # prepare_time_axis needs an in-memory AnnData; obs is already in-memory
    # even for backed files, so we operate on a metadata-only copy.
    meta = _ad.AnnData(obs=adata.obs.copy(), var=adata.var.copy())
    meta, all_times, model_to_day = prepare_time_axis(meta)

    # Write the prepared columns back into the backed adata's obs (obs is
    # a pandas DataFrame in-memory even for backed files).
    for col in ['time', 'day', 'stage_value']:
        if col in meta.obs.columns:
            adata.obs[col] = meta.obs[col].values

    adata.uns['gene_names']   = gene_names_from_var(adata.var)
    adata.uns['all_times']    = all_times
    adata.uns['model_to_day'] = model_to_day

    return adata, all_times, model_to_day
