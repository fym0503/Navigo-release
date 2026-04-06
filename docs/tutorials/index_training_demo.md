# Training Demo

This section walks through a compact **Navigo training and validation workflow** on a sampled mouse embryogenesis subset.

The goal is to show the end-to-end training path on a tutorial-scale dataset: subset sampling, GPU training with `submission/main_navigo.py`, and held-out intermediate-time validation with shared-embedding UMAP panels.

```{admonition} Runtime Note
:class: note
The notebook is designed for the Navigo package environment, expects a CUDA-capable PyTorch setup for GPU training, and uses the repository-local interpolation atlas under `data/`.
```

```{toctree}
:maxdepth: 1

notebooks/training_demo/01_Navigo_Training_Demo_executed
```
