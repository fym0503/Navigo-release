# Navigo-release

<p align="center">
  <img src="docs/logo_navigo.png" alt="Navigo logo" style="display:block; width:100%; max-width:100%;">
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
git clone https://github.com/fym0503/Navigo-release.git
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

The main tutorial notebooks are organized by biological application under [`docs/tutorials/notebooks`](docs/tutorials/notebooks):

- [`grn_section`](docs/tutorials/notebooks/grn_section): regulatory analysis for congenital heart disease.
- [`knockout_section`](docs/tutorials/notebooks/knockout_section): zero-shot perturbation prediction and compensation analysis.
- [`interpolation_section`](docs/tutorials/notebooks/interpolation_section): interpolation and denoising along developmental trajectories.
- [`reprogramming_section`](docs/tutorials/notebooks/reprogramming_section): cardiac and neuronal reprogramming studies.

Supporting resources and legacy analysis scripts are under [`docs/tutorials/resources`](docs/tutorials/resources).

Some notebooks depend on local datasets and pretrained checkpoints stored under `data/` and `checkpoints/`.

## Repository Structure

- `navigo/`: **installable Python package**.
- `docs/`: Sphinx/MyST documentation source.
- `docs/tutorials/notebooks/`: end-to-end tutorial notebooks.
- `docs/tutorials/resources/`: helper scripts, reference assets, and legacy analysis materials.
- `submission/`: submission-time entrypoints and helper scripts.

## Citation

If you use this code, please cite:

```bibtex
@misc{fan_navigo,
  title={Generative Modeling of Mouse Embryogenesis for Fate and Disease Prediction},
  author={Yimin Fan and Xinyuan Liu and Yixuan Wang and Zehua Zeng and Lei Li and Xiaojie Qiu and Yu Li}
}
```
