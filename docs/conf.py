# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import hyperpyyaml


sys.path.insert(0, os.path.abspath("../speechbrain"))


# -- Project information -----------------------------------------------------

project = "SpeechBrain"
copyright = "2021, SpeechBrain"
author = "SpeechBrain"

# The full version, including alpha/beta/rc tags
release = "0.5.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "recommonmark",
]


# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping:
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
}

# AUTODOC:

autodoc_default_options = {}

# Autodoc mock extra dependencies:
autodoc_mock_imports = ["numba", "sklearn"]

# Order of API items:
autodoc_member_order = "bysource"
autodoc_default_options = {"member-order": "bysource"}

# Don't show inherited docstrings:
autodoc_inherit_docstrings = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_apidoc_templates"]

# -- Better apidoc -----------------------------------------------------------


def run_apidoc(app):
    """Generage API documentation"""
    import better_apidoc

    better_apidoc.APP = app

    better_apidoc.main(
        [
            "better-apidoc",
            "-t",
            "_apidoc_templates",
            "--force",
            "--no-toc",
            "--separate",
            "-o",
            "API",
            os.path.dirname(hyperpyyaml.__file__),
        ]
    )
    better_apidoc.main(
        [
            "better-apidoc",
            "-t",
            "_apidoc_templates",
            "--force",
            "--no-toc",
            "--separate",
            "-o",
            "API",
            os.path.join("../", "speechbrain"),
        ]
    )


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# See https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
# for rtd theme options
html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}


def setup(app):
    app.connect("builder-inited", run_apidoc)
