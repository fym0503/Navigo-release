# Navigo-release

<p align="center">
  <img src="https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/navigo_logo.png" alt="Navigo logo" style="display:block; width:100%; max-width:100%;">
</p>

<p align="center">
  Generative Modeling of Mouse Embryogenesis for Fate and Disease Prediction
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#highlights">Highlights</a> |
  <a href="#installation">Installation</a> |
  <a href="#tutorials">Tutorials</a> |
  <a href="#repository-structure">Repository Structure</a> |
  <a href="#citation">Citation</a>
</p>

## Overview

Navigo is a **generative modeling framework for developmental biology**. It combines population-level flow matching with molecular-level RNA kinetics to learn continuous developmental vector fields from single-cell transcriptomic data.

In this repository, Navigo is packaged together with **tutorial notebooks and analysis resources** used for:

- developmental trajectory interpolation and denoising;
- congenital heart disease regulatory analysis;
- zero-shot prediction of genetic perturbation effects;
- fibroblast reprogramming analysis and transcription factor screening.

Navigo is designed for modeling **embryogenesis across large temporal single-cell atlases** and for supporting in silico perturbation and reprogramming studies.

## Highlights

<p align="center">
  <img src="docs/navigo_illu.png" alt="Navigo overall illustration" style="display:block; width:100%; max-width:100%;">
</p>

- **Learns developmental dynamics** from temporal single-cell snapshots with a continuous vector field formulation.
- Couples flow matching with transcriptional kinetics to keep the learned dynamics biologically grounded.
- **Supports counterfactual perturbation analysis** without perturbation-specific training.
- Enables regulatory network analysis for developmental disease settings such as congenital heart defects.
- Supports screening of reprogramming interventions, including inhibitory factors for cardiac conversion and synergistic transcription factor combinations for neuronal conversion.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/aristoteleo/Navigo-release.git
cd Navigo-release
pip install -r requirements.txt
pip install -e .
```

To build the documentation locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Tutorials

The tutorial site is organized into five sections:

- [`Training Demo`](docs/tutorials/index_training_demo.md): compact end-to-end subset training and validation workflow.
- [`Interpolation`](docs/tutorials/index_interpolation.md): held-out interpolation benchmarking and denoising examples.
- [`GRN`](docs/tutorials/index_grn.md): congenital heart disease regulatory analysis workflow.
- [`Knockout`](docs/tutorials/index_knockout.md): zero-shot perturbation prediction and enrichment summaries.
- [`Reprogramming`](docs/tutorials/index_reprogramming.md): cardiac and neuronal reprogramming studies.

The underlying notebooks live under [`docs/tutorials/notebooks`](docs/tutorials/notebooks), and supporting resources remain under [`docs/tutorials/resources`](docs/tutorials/resources).

To run the tutorials locally, download the tutorial asset bundles referenced in [`docs/tutorials/index.md`](docs/tutorials/index.md), then extract them at the repository root so `data/` and `checkpoints/` are available alongside `docs/` and `navigo/`.

## Repository Structure

- `navigo/`: installable Python package with the core modeling and analysis code.
- `docs/`: Sphinx/MyST documentation source for the main site and tutorial pages.
- `docs/tutorials/notebooks/`: end-to-end tutorial notebooks used throughout the documentation.
- `data/` and `checkpoints/`: local tutorial assets expected by the notebooks and training demo.
- `submission/`: command-line entrypoints and helper scripts, including the docs sync utility.

## Citation

If you use this code, please cite:

```bibtex
@article{fan_navigo_2026,
  title={Generative Modeling of Mouse Embryogenesis for Fate and Disease Prediction},
  author={Yimin Fan and Xinyuan Liu and Yixuan Wang and Zehua Zeng and Lei Li and Xiaojie Qiu and Yu Li},
  year={2026}
}
```
