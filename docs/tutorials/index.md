# Tutorials

The easiest way to get familiar with Navigo is to follow along with our tutorials.
Many are also designed to work seamlessly in Google Colab, a free cloud computing platform.
In this unified repository, tutorial notebooks live under `docs/tutorials/notebooks/`, while shared inputs are centralized at the repository root in `data/` and `checkpoints/`.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Interpolation
:link: index_interpolation
:link-type: doc

Myofibroblast temporal interpolation and denoising analysis across proportion, marker, and pathway readouts.
Held-out interpolation benchmarking and myofibroblast denoising analysis across quantitative and figure-level outputs.
:::

:::{grid-item-card} Training Demo
:link: index_training_demo
:link-type: doc

Subset-scale GPU training and held-out intermediate validation for the full Navigo training workflow.
:::

:::{grid-item-card} GRN
:link: index_grn
:link-type: doc

CHD-focused regulatory program analysis based on knockout response organization and pathway interpretation.
:::

:::{grid-item-card} Knockout
:link: index_knockout
:link-type: doc

Lineage-resolved knockout analyses spanning pathway enrichment, severity comparison, and directional enrichment summaries.
:::

:::{grid-item-card} Reprogramming
:link: index_reprogramming
:link-type: doc

Cardiac and neuronal reprogramming analyses driven by in silico perturbation and ranking.
:::
::::

```{toctree}
:maxdepth: 2

index_interpolation
index_training_demo
index_grn
index_knockout
index_reprogramming

```
