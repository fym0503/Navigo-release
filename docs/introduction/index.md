# Introduction

**Navigo is a generative model for developmental single-cell transcriptomics** that learns a **continuous developmental vector field** from temporal scRNA-seq atlases by combining **population-level flow matching** with **molecular-level RNA kinetics**.

This learned field can then be queried for **trajectory mapping**, **temporal interpolation**, **denoising**, **perturbation simulation**, and **regulatory analysis**. In practice, Navigo supports applications ranging from cardiac GRN interpretation and zero-shot knockout analysis to fibroblast reprogramming.

The method is built around two core ideas:

1. Development should be modeled at the **molecular level** using RNA kinetics, so the learned velocity field is tied to transcription, splicing, and degradation rather than an arbitrary black-box transition function.
2. Development should also be modeled at the **population level**, where the system must learn coherent transitions between time points even though destructive single-cell measurements do not provide true cell-to-cell correspondences across time.

This section introduces the main technical components of the model.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Model Principles
:link: model
:link-type: doc

How Navigo represents cell state and the developmental vector field.
:::

:::{grid-item-card} Training
:link: training
:link-type: doc

How the model is trained.
:::

:::{grid-item-card} Inference
:link: inference
:link-type: doc

How the learned field is used at inference time.
:::

:::{grid-item-card} Perturbation and GRN Analysis
:link: perturbation
:link-type: doc

How Navigo supports perturbation and GRN analysis.
:::
::::
