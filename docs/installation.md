# Installation

## Quick install

Navigo can be installed directly from this repository. We recommend using a virtual environment.


```bash
git clone https://github.com/aristoteleo/Navigo-release.git Navigo-release
cd Navigo-release
pip install -r requirements.txt
pip install -e .
```

## Build the documentation locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
