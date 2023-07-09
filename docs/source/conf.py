import datetime
import os
import shutil
import sys

import tomli

# -- Project information
sys.path.insert(0, os.path.abspath("../../"))

now = datetime.date.today()

project = "PyVRP"
authors = "PyVRP contributors"
copyright = f"2022 - {now.year}, {authors}"

with open("../../pyproject.toml", "rb") as fh:
    pyproj = tomli.load(fh)
    release = version = pyproj["tool"]["poetry"]["version"]

print("Copying example notebooks into docs/source/examples/")
shutil.copytree("../../examples", "examples/", dirs_exist_ok=True)

# -- API documentation
autoapi_type = "python"
autoapi_dirs = ["../../pyvrp"]
autoapi_options = ["members", "undoc-members", "special-members"]
autoapi_ignore = ["*/tests/*.py", "*/cli.py"]

autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_add_objects_to_toctree = False

autoapi_python_class_content = "class"
autoapi_member_order = "bysource"
autodoc_typehints = "signature"

# -- numpydoc
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
napoleon_include_special_with_doc = True

# -- nbsphinx
nbsphinx_execute = ["always"]

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "nbsphinx",
    "numpydoc",
    "sphinxcontrib.spelling",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

add_module_names = False

# -- Options for HTML output
html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
