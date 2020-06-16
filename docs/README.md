# SpeechBrain documentation

Run

    make html

to build HTML documentation. Then open `build/html/index.html`

## Automatic API documentation from docstrings

The documentation uses `sphinx.ext.napoleon` to support Google-style
docstrings. Sphinx natively supports reStructuredText directives.

Automatically generating documentation based on docstrings is not the
core of Sphinx. Infact it relies on `sphinx-apidoc` tool.

## Future work

Besides automatic API documentation, Sphinx will facilitate manual prose
documentation.
