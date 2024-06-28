import datetime
import importlib
import inspect
import os
import shutil
import subprocess
import sys
from typing import Optional

import tomli

# -- Project information
sys.path.insert(0, os.path.abspath("../../"))

now = datetime.date.today()

project = "PyVRP"
authors = "PyVRP contributors"
repo_url = "https://github.com/PyVRP/PyVRP/"
copyright = f"2022 - {now.year}, {authors}"

with open("../../pyproject.toml", "rb") as fh:
    pyproj = tomli.load(fh)
    release = version = pyproj["tool"]["poetry"]["version"]

print("Copying example notebooks into docs/source/examples/")
shutil.copytree("../../examples", "examples/", dirs_exist_ok=True)

# -- API documentation
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True


# -- sphinx.ext.linkcode
def linkcode_resolve(domain: str, info: dict) -> Optional[str]:
    """
    Generates a URL pointing to the source code of a specified object located
    in the PyVRP repository.

    Parameters
    ----------
    domain: str
        The domain of the object (e.g., "py" for Python, "cpp" for C++).
    info: dict
        A dictionary with keys "module" and "fullname". "module" contains
        the module name as a string, and "fullname" contains the full object
        name as a string.

    Returns
    -------
    Optional[str]
        URL pointing to the identified object's source code in the PyVRP
        repository, if the object can be identified.
    """
    if domain != "py" or not info.get("module") or not info.get("fullname"):
        return None

    obj = importlib.import_module(info["module"])
    for attr_name in info["fullname"].split("."):
        obj = getattr(obj, attr_name)

    try:
        source = inspect.getsourcefile(obj)
    except TypeError:
        return None

    if source is None:
        # This is one of the native extensions, which we cannot yet handle.
        return None

    rel_path = source[source.rfind("pyvrp/") :]
    line_num = inspect.getsourcelines(obj)[1]

    cmd = "git rev-parse --short HEAD".split()
    revision = subprocess.check_output(cmd).strip().decode("ascii")

    return f"{repo_url}/blob/{revision}/{rel_path}#L{line_num}"


# -- numpydoc
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
napoleon_include_special_with_doc = True

# -- nbsphinx
skip_notebooks = os.getenv("SKIP_NOTEBOOKS", False)
nbsphinx_execute = "never" if skip_notebooks else "always"

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    "nbsphinx",
    "numpydoc",
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
python_use_unqualified_type_names = True

# -- Options for HTML output
html_theme = "sphinx_immaterial"
html_logo = "assets/images/icon.svg"
html_theme_options = {
    "site_url": "https://pyvrp.org/",
    "repo_url": repo_url,
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "features": [
        "navigation.expand",
        "navigation.top",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "primary": "orange",
            "accent": "yellow",
            "scheme": "default",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "primary": "orange",
            "accent": "yellow",
            "scheme": "slate",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": True,
    "version_info": [  # TODO: move this out of the configuration file
        {
            "version": "",
            "title": "Development",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.9.0",
            "title": "v0.9.0",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.8.1",
            "title": "v0.8.1",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.7.0",
            "title": "v0.7.0",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.6.0",
            "title": "v0.6.0",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.5.0",
            "title": "v0.5.0",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.4.4",
            "title": "v0.4.4",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.3.0",
            "title": "v0.3.0",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.2.1",
            "title": "v0.2.1",
            "aliases": [],
        },
        {
            "version": "https://pyvrp.github.io/v0.1.0",
            "title": "v0.1.0",
            "aliases": [],
        },
    ],
}

python_resolve_unqualified_typing = True
python_transform_type_annotations_pep585 = True
python_transform_type_annotations_pep604 = True
object_description_options = [
    ("py:.*", dict(include_fields_in_toc=False, include_rubrics_in_toc=False)),
    ("py:attribute", dict(include_in_toc=False)),
    ("py:parameter", dict(include_in_toc=False)),
]


# -- Options for EPUB output
epub_show_urls = "footnote"
