# Where is what, a link list.
```
                  (documentation)           (tutorials)
                  .—————————————.            .———————.
                  | readthedocs |       ‚––> | Colab |
                  \—————————————/      ∕     \———————/
                         ^       ‚––––‘          |
    (release)            |      ∕                v
    .——————.       .———————————. (landing) .———————————.
    | PyPI | –––>  | github.io |  (page)   | templates |   (reference)
    \——————/       \———————————/       ‚–> \———————————/ (implementation)
        |                |        ‚–––‘          |
        v                v       ∕               v
.———————————–—.   .———————————–—.           .—————————.           .~~~~~~~~~~~~~.
| HyperPyYAML |~~~| speechbrain | ––––––––> | recipes | ––––––––> | HuggingFace |
\————————————–/   \————————————–/           \—————————/     ∕     \~~~~~~~~~~~~~/
  (usability)     (source/modules)          (use cases)    ∕    (pretrained models)
                                                          ∕
                        |                        |       ∕               |
                        v                        v      ∕                v
                  .~~~~~~~~~~~~~.            .~~~~~~~~.            .———————————.
                  |   PyTorch   | ––––––––-> | GDrive |            | Inference |
                  \~~~~~~~~~~~~~/            \~~~~~~~~/            \———————————/
                   (checkpoints)             (results)            (code snippets)
```

<br/>What is where?

1. https://speechbrain.github.io/
  <br/> a. via: https://github.com/speechbrain/speechbrain.github.io
  <br/> b. pointing to several tutorials on Google Colab
  <br/> `python & yaml`
2. https://github.com/speechbrain/speechbrain
  <br/> a. [docs](https://github.com/speechbrain/speechbrain/tree/develop/docs) for https://speechbrain.readthedocs.io/
  <br/> b. [recipes](https://github.com/speechbrain/speechbrain/tree/develop/recipes)
  <br/>`python & yaml & README`
  <br/> c. [speechbrain](https://github.com/speechbrain/speechbrain/tree/develop/speechbrain), heavily tied with [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML); released on [PyPI](https://pypi.org/project/speechbrain/)
  <br/>`python & yaml`
  <br/> d. [templates](https://github.com/speechbrain/speechbrain/tree/develop/templates)
  <br/>`python & yaml & README`
  <br/> e. [tools](https://github.com/speechbrain/speechbrain/tree/develop/tools) for non-core functionality
  <br/>`perl; python & yaml`
3. https://huggingface.co/speechbrain/
  <br/> hosting several model cards (pretrained models with code snippets)
  <br/> `python & yaml`
  <br/> [option to host datasets]
4. Gdrive (and alike)
  <br/> hosting training results; checkpoints; ...

These points need testing coverage.