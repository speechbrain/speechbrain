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

## Tutorial integration

Tutorials are now inside of the main SpeechBrain repository.

### Contributor guidelines for tutorials

The `docs/tutorials` directory exclusively contains tutorials in Jupyter Notebook format. These tutorials are integrated into the doc semi-automatically. You should ensure that the following steps are respected so that they render correctly and so that we can keep consistent quality.

#### Relatively important notices

- Create your new notebook, preferably with the same structure as existing tutorials.
- Keep the file size low! Limit images and audio.
  - Ideally it should be a few hundred KiB total, avoid anything larger than 1MiB unless you really have to.
  - It's OK if the user has to run the notebook to get some of the heavier outputs.
- Preferably use Jupyter Notebook for final editing of your notebook.
  - Jupyter Notebook tends to have somewhat sane `.ipynb` output. This avoids Git diffs from being excessively large.
- **Images can be put in the `docs/tutorials/assets` directory,** rather than embedded as base64. You can then refer to them in Markdown like `![alt text](../assets/myimage.png)`. These will work correctly when imported on Colab.
  - Pick descriptive names.

#### Integration in documentation

- Add your notebook to the relevant category `.rst`, paying attention to keep the same structure and appearance as existing tutorials.
  - (Create a category if _really_ necessary, but this bloats the table of contents/sidebar.)
- **The Colab header/citation footer** are generated automatically and should not be manually inserted or edited. See `tools/tutorial-cell-update.py`. You should run this for your notebook.
- Add your notebook to the hidden `toctree` of the same document.
- Make sure that your headings are consistent!
  - Please use a single top-level heading for the title of your notebook.
  - That title should match the name in the summary.
  - Please use level-2 or deeper headings for everything else (`##`, `###`, etc. in markdown). Notebook headings **are** used as part of the document tree!
- Make sure that your tutorial renders at least correctly with the in-documentation view.
  - You can check this by either generating docs normally, or use the readthedocs PR integration that lets you preview docs for your PR (assuming it succeeds). This takes time, though! You preferably really should have a functional documentation environment when contributing.