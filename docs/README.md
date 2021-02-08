# SpeechBrain documentation

Please install additional dependencies:

```
pip install -r docs-requirements.txt
```

Then run:
```
make html
```
to build HTML documentation. Then open `build/html/index.html`

## Automatic API documentation from docstrings

The documentation uses `sphinx.ext.napoleon` to support Google-style
docstrings. Sphinx natively supports reStructuredText directives.

Automatically generating documentation based on docstrings is not the
core of Sphinx. For this, after much searching, we use better-apidoc.

## Future work

Besides automatic API documentation, Sphinx will facilitate manual prose
documentation.
