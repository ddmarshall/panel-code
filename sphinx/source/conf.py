# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyPC'
copyright = '2022, David D. Marshall'
author = 'David D. Marshall'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'numpydoc',
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.imgmath'
]

napoleon_google_docstring = False
autosummary_generate = True
html_show_sourcelink = False
autodoc_inherit_docstrings = True
add_module_names = False
numpydoc_class_members_toctree = False
imgmath_image_format = 'svg'
imgmath_font_size = 14

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
#html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
