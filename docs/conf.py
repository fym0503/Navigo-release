from datetime import datetime
from pathlib import Path
import sys
import types

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _install_doc_mocks() -> None:
    """Install lightweight dependency stubs for API-doc builds."""

    class _DummyLayer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _NoGrad:
        def __call__(self, func=None):
            if func is None:
                return self
            return func

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyAnnData:
        def __init__(self, *args, **kwargs):
            self.obs = kwargs.get("obs")
            self.var = kwargs.get("var")
            self.uns = kwargs.get("uns", {})
            self.layers = kwargs.get("layers", {})
            self.obsm = kwargs.get("obsm", {})

    torch_mod = types.ModuleType("torch")
    torch_nn_mod = types.ModuleType("torch.nn")
    torch_nn_func_mod = types.ModuleType("torch.nn.functional")
    torch_utils_mod = types.ModuleType("torch.utils")
    torch_utils_data_mod = types.ModuleType("torch.utils.data")
    anndata_mod = types.ModuleType("anndata")

    torch_mod.Tensor = object
    torch_mod.float32 = "float32"
    torch_mod.no_grad = _NoGrad()
    torch_mod.manual_seed = lambda *args, **kwargs: None
    torch_mod.load = lambda *args, **kwargs: {}
    torch_mod.randn = lambda *args, **kwargs: None
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *args, **kwargs: None,
    )

    torch_nn_mod.Module = object
    torch_nn_mod.Linear = _DummyLayer
    torch_nn_mod.Parameter = lambda value: value
    torch_nn_func_mod.relu = lambda value, *args, **kwargs: value
    torch_nn_func_mod.sigmoid = lambda value, *args, **kwargs: value
    torch_utils_data_mod.Dataset = object
    torch_utils_data_mod.DataLoader = object
    torch_utils_mod.data = torch_utils_data_mod

    anndata_mod.AnnData = _DummyAnnData
    anndata_mod.read_h5ad = lambda *args, **kwargs: _DummyAnnData()
    anndata_mod.read = anndata_mod.read_h5ad

    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.nn", torch_nn_mod)
    sys.modules.setdefault("torch.nn.functional", torch_nn_func_mod)
    sys.modules.setdefault("torch.utils", torch_utils_mod)
    sys.modules.setdefault("torch.utils.data", torch_utils_data_mod)
    sys.modules.setdefault("anndata", anndata_mod)
    sys.modules.setdefault("ot", types.SimpleNamespace(emd2=lambda *args, **kwargs: 0.0))
    sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))
    sys.modules.setdefault("umap", types.SimpleNamespace(UMAP=object))
    sys.modules.setdefault("adjustText", types.SimpleNamespace(adjust_text=lambda *args, **kwargs: None))


_install_doc_mocks()

# -- Project information -----------------------------------------------------
project = "Navigo"
author = "Navigo Team"
copyright = f"{datetime.now():%Y}, {author}."
version = "main"
release = "main"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "substitution",
]
myst_url_schemes = ("http", "https", "mailto")

autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"

nb_execution_mode = "off"
nb_merge_streams = True
nb_output_stderr = "remove"

bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = project
html_logo = "https://raw.githubusercontent.com/Starlitnightly/ImageStore/main/omicverse_img/navigo_logo.png"
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["css/override.css"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb4ff",
        "color-brand-content": "#7fb4ff",
    },
}

ogp_social_cards = {"enable": False}

pygments_style = "tango"
pygments_dark_style = "monokai"
