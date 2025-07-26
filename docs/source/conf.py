# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions or modules are in another directory,
# add these directories to sys.path here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NNMD"
copyright = "2025, Andrey Budnikov"
author = "Andrey Budnikov"
release = "0.2.0-dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = ['.rst', '.md']

language = "en"  # Set the default language for the documentation
supported_languages = {
    "en": "NNMD %s documentation in English",
    "ru": "Документация NNMD %s на русском языке",
}
locale_dirs = ["./../locale/"]  # path is example but recommended.
gettext_compact = False

# Figures are enumerated and can be reference by the :numref: directive
numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "../images/nnmd_logo.png"
html_theme_options = {
    "repository_url": "https://github.com/ded-otshelnik/nnmd",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_download_button": False,
}
html_context = {
    "current_version": release,
    "versions": [[release, f"link to {release}"]],
    "current_language": "en",
    "languages": [["en", "link to en"], ["ru", "link to ru"]],
}

# html_static_path = ["_static"]

# -- Options for References and Bibliography -----------------------------
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/configuration
bibtex_bibfiles = ["refs.bib"]
bibtex_encoding = "utf-8-sig"
bibtex_default_style = "unsrt"
bibtex_reference_style = "label"
