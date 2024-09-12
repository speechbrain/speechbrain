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

It seems better-apidoc doesn't use autodoc\_mock\_imports so we currently just
add all extra dependencies to docs-requirements.txt

## Tutorial integration

Tutorials are now inside of the main SpeechBrain repository.

### Contributor guidelines for tutorials

- Create your new notebook, preferably with the same structure as existing tutorials.
- Add your notebook to the relevant category `.rst`, paying attention to keep the same structure and appearance as existing tutorials.
  - (Create a category if _really_ necessary, but this bloats the table of contents/sidebar.)
- Add your notebook to the hidden `toctree` of the same document.
- Make sure that your headings are consistent!
  - Please use a single top-level heading for the title of your notebook.
  - That title should match the name in the summary.
  - Please use level-2 or deeper headings for everything else (`##`, `###`, etc. in markdown). Notebook headings **are** used as part of the document tree!
- Make sure that your tutorial renders at least correctly with the in-documentation view.
  - You can check this by either generating docs normally, or use the readthedocs PR integration that lets you preview docs for your PR (assuming it succeeds). This takes time, though! You preferably really should have a functional documentation environment when contributing.