# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

from pkg_resources import get_distribution

sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())

project = "legend-daq2lh5"
copyright = "2023, the LEGEND Collaboration"
version = get_distribution("legend-daq2lh5").version

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# Furo theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/legend-exp/legend-daq2lh5",
    "source_branch": "main",
    "source_directory": "docs/source",
}
html_title = f"{project} {version}"

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_custom_sections = ["JSON Configuration Example"]

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "lgdo": ("https://legend-pydataobj.readthedocs.io/en/stable", None),
    "dspeed": ("https://dspeed.readthedocs.io/en/stable", None),
}  # add new intersphinx mappings here

# sphinx-autodoc
autodoc_default_options = {"ignore-module-all": True}
# Include __init__() docstring in class docstring
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"
