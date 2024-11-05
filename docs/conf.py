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

import better_apidoc
import hyperpyyaml
from sphinx.ext.autodoc.mock import mock

sys.path.insert(-1, os.path.abspath("../"))


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
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_markdown_tables",
    "recommonmark",
    # chose myst-nb over nbsphinx is annoying because of the pandoc dependency
    # of the latter, which needs to be installed system-wide or through conda
    "myst_nb",
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
    "torchaudio": ("https://pytorch.org/audio/stable/", None),
}

# Myst-NB documentation

jupyter_execute_notebooks = "off"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# AUTODOC:

autodoc_default_options = {}

# Autodoc mock extra dependencies -- doesn't work out of the box, because of better_apidoc.
#
# So, let's reuse the autodoc mock...
#
# We would also like to mock more imports than this but this is shockingly prone
# to randomly breaking, so let's keep a small-ish set of dependencies that tend
# to be more annoying to install and to nuke our CI on update
autodoc_mock_imports = [
    "k2",
    "flair",
    "fairseq",
    "spacy",
    "ctc_segmentation",
]

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
exclude_patterns = ["_apidoc_templates", "build"]

# Make backticks behave as inline code blocks rather than italics
default_role = "code"

# -- Better apidoc -----------------------------------------------------------


def run_apidoc(app):
    """Generate API documentation"""

    with mock(autodoc_mock_imports):
        try:
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
                    os.path.join("../", "speechbrain"),
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
                    os.path.dirname(hyperpyyaml.__file__),
                ]
            )
        except Exception:
            # because otherwise sphinx very helpfully eats the backtrace
            import traceback

            print(traceback.format_exc(), file=sys.stderr)
            raise


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# See https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
# for rtd theme options
html_theme_options = {
    "logo_only": True,
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
}

html_logo = "images/speechbrain-logo.svg"


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
