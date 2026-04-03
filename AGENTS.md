# AGENTS.md

SpeechBrain is an open-source PyTorch toolkit for conversational AI (speech recognition, speaker verification, speech enhancement, separation, TTS, spoken language understanding, and more) known for its ease of use and flexibility. It is Apache 2.0 licensed.

## Project structure

```
speechbrain/          # Core library (importable as `import speechbrain`)
  core.py             # Brain class — the central training/eval orchestrator
  dataio/             # Data loading, batching, samplers, dataset objects
  nnet/               # Neural network building blocks (RNN, CNN, attention, transformers, losses, etc.)
  lobes/              # Higher-level model components (feature extractors, encoders, full models like CRDNN, wav2vec2)
  decoders/           # CTC, seq2seq beam search, transducer decoders
  processing/         # Signal processing (features, augmentation, multi-mic)
  utils/              # Checkpointing, distributed training, metrics, logging, profiling
  inference/          # Pretrained model interfaces (EncoderClassifier, EncoderDecoderASR, etc.)
  integrations/       # Optional heavy-dependency integrations (Transformers, Whisper, etc.)
  lm/                 # Language model utilities
recipes/              # Training scripts organized as recipes/{dataset}/{task}/{model}/
  {dataset}/{task}/
    {model}/          # Recipe implementation for a specific model/configuration
      train.py        # Training script (subclasses Brain)
      hparams/        # HyperPyYAML config files (train.yaml, etc.)
      extra-requirements.txt  # Recipe-specific pip dependencies (if any)
      README.md       # Results, how to run, pretrained model links
templates/            # Minimal working examples to bootstrap new recipes
tests/
  unittests/          # Unit tests for core library
  integration/        # Integration tests (small end-to-end training runs)
docs/                 # Documentation 
  tutorials/          # Jupyter notebooks integrated into ReadTheDocs
tools/                # Maintenance scripts (tutorial cell updater, etc.)
```

## Architecture concepts

### The Brain class (`speechbrain.core.Brain`)

Brain is the central abstraction for all training and evaluation. Every recipe subclasses it and overrides the following methods:

- `compute_forward(batch, stage)` — forward pass, returns predictions
- `compute_objectives(predictions, batch, stage)` — computes loss, logs metrics

The `stage` argument is a `Stage` enum: `TRAIN`, `VALID`, or `TEST` which defines the current stage of the training loop. Brain handles the training loop, checkpointing, distributed training (DDP), gradient accumulation, mixed precision, and logging.

```python
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        wavs, lens = batch.sig  # (batch, time), (batch,) relative lengths
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.encoder(feats)
        return self.modules.decoder(feats)

    def compute_objectives(self, predictions, batch, stage):
        tokens, token_lens = batch.tokens
        loss = self.hparams.ctc_cost(predictions, tokens, lens, token_lens)
        if stage != sb.Stage.TRAIN:
            self.cer_metric.append(batch.id, predictions, tokens)
        return loss
```

Key lifecycle methods you can override: `on_stage_start`, `on_stage_end`, `on_fit_batch_end`, `fit_batch`, `evaluate_batch`, `init_optimizers`.

### HyperPyYAML (the YAML config system)

SpeechBrain uses HyperPyYAML, an extended YAML syntax maintained by SpeechBrain at https://github.com/speechbrain/HyperPyYAML. This is NOT plain YAML — it is a declarative system that can instantiate Python objects, resolve references, and perform simple arithmetic. Understanding it is essential.

Key tags:
- `!new:module.ClassName` — instantiates a Python object. Indented keys become constructor kwargs.
- `!name:module.ClassName` — returns a callable (like `functools.partial`) without calling it. Used for optimizers, schedulers.
- `!ref <key>` — references another value in the YAML. Supports `<key[subkey]>`, string interpolation (`<folder>/subdir`), and arithmetic (`<foo> * 2 + 1`).
- `!copy <key>` — deep-copies the referenced object (vs `!ref` which is a shallow reference).
- `!apply:module.function` — calls the function immediately during YAML loading.
- `!tuple` — creates a Python tuple.
- `!PLACEHOLDER` — errors at load time, must be overridden via CLI or overrides dict.

Example pattern from a real recipe:
```yaml
seed: 1234
output_folder: !ref results/asr/<seed>
save_folder: !ref <output_folder>/save

model: !new:speechbrain.lobes.models.CRDNN.CRDNN
    output_size: 40
    cnn_blocks: 2
    dnn_blocks: 2

opt_class: !name:torch.optim.Adam
    lr: 0.001

epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounter
    limit: !ref <number_of_epochs>
```

**Critical: loading YAML executes arbitrary Python code** — `!new:` will import and instantiate anything. Treat YAML files with the same caution as Python code.

Overrides from CLI:
```bash
python train.py hparams/train.yaml --seed 42 --lr 0.0001 --data_folder /path/to/data --num_epochs=100
```

### Tensor convention

All tensors follow **batch-time-channels** ordering:
- `(batch, time)` for raw waveforms
- `(batch, time, channels)` for features
- `(batch, time, channels, ...)` for multi-channel

Lengths are tracked as **relative lengths** (0.0 to 1.0), representing the fraction of the max length in the batch. This avoids passing absolute lengths and simplifies padding/masking. Example: a batch of 3 signals with lengths [16000, 12000, 8000] has relative lengths [1.0, 0.75, 0.5].

### Data pipeline
 
The pipeline has three layers: **data manifests** (JSON/CSV) → **DynamicItemDataset** → **Dynamic Item Pipelines**.
 
**Manifests** are JSON or CSV files containing static items (file paths, transcriptions, speaker IDs, durations). JSON format: `{"utt1": {"wav": "path.flac", "wrd": "HELLO", "spk_id": "spk01", "duration": 3.5}, ...}`. Each recipe provides a preparation script that parses raw datasets into this format. Manifests must include a `duration` field for dynamic batching to work.
 
**DynamicItemDataset** (`speechbrain.dataio.dataset`) loads a manifest and supports on-the-fly transformations via dynamic items. Dependencies between items are resolved automatically as a DAG. Items are evaluated **lazily** — only items in `set_output_keys` (and their dependencies) are computed.
 
```python
train_data = DynamicItemDataset.from_csv(csv_path=hparams["train_csv"],
    replacements={"data_root": hparams["data_folder"]})  # replacements substitute placeholders in manifest values
```
 
**Dynamic Item Pipelines** are functions decorated with `@sb.utils.data_pipeline.takes(...)` / `@sb.utils.data_pipeline.provides(...)`.
 
```python
@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(wav):
    return sb.dataio.dataio.read_audio(wav)
 
@sb.utils.data_pipeline.takes("wrd")
@sb.utils.data_pipeline.provides("wrd", "tokens_bos", "tokens_eos", "tokens")
def text_pipeline(wrd):
    yield wrd
    tokens_list = tokenizer.encode_as_ids(wrd)
    yield torch.LongTensor([bos] + tokens_list)   # tokens_bos
    yield torch.LongTensor(tokens_list + [eos])    # tokens_eos
    yield torch.LongTensor(tokens_list)            # tokens
```
 
Register pipelines and declare outputs — typically applied to all splits at once:
 
```python
datasets = [train_data, valid_data, test_data]
sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "tokens_bos", "tokens_eos", "tokens"])
```
 
**Filtering and sorting**: `train_data.filtered_sorted(sort_key="duration")` returns a sorted view (shared static data, no copy). Supports `key_min_value`, `key_max_value`, `select_n`. When sorting, disable dataloader shuffle or sorting is pointless.
 
**PaddedBatch** (`speechbrain.dataio.batch`) is the collate function. It pads variable-length tensors and returns `PaddedData(data, lengths)` namedtuples. Always unpack: `wavs, wav_lens = batch.sig`. The `lengths` are relative (0.0–1.0). Use `SaveableDataLoader` instead of raw `DataLoader` — it supports checkpoint-resumable iteration.
 
**DynamicBatchSampler** (`speechbrain.dataio.sampler`) groups utterances into length-bucketed batches with a target total duration instead of a fixed batch size. Requires a `length_func` pointing to the manifest duration field. Do not pass `batch_size` when using `batch_sampler`.
 
**CategoricalEncoder** (`speechbrain.dataio.encoder`) maps string labels to integer indices for classification. Fit with `encoder.update_from_didataset(train_data, "spk_id")`, use `encoder.encode_label_torch(label)` inside a pipeline, sanity-check with `encoder.expect_len(num_classes)`.
 
Every recipe wires this together in a `dataio_prep(hparams)` function — follow this pattern for new recipes.
 
## Recipe conventions
 
Every recipe lives at `recipes/{dataset}/{task}/` and follows this structure:
 
- `train.py` — the training script. It subclasses `Brain`, defines the dataio pipeline, and calls `brain.fit()` / `brain.evaluate()`.
- `hparams/*.yaml` — HyperPyYAML files. There may be multiple configs for different model variants.
- `README.md` — documents results (WER, EER, etc.), how to run, and links to pretrained models on HuggingFace.
- `extra-requirements.txt` — if the recipe needs packages not in core requirements.
 
To run a recipe:
```bash
cd recipes/{dataset}/{task}
python train.py hparams/train.yaml --data_folder /path/to/data
```
 
When creating a new recipe, start from `templates/` for a minimal working skeleton.
 
## Running tests

```bash
# Run pre-commit checks (formatting, linting via ruff)
pre-commit run -a

# Run doctests
pytest --doctest-modules speechbrain/path/to/module.py

# Run unit tests
pytest tests/unit/

# Run a specific integration test
pytest tests/integration/ASR_CTC/ -xvs

# Run all integration tests
pytest tests/integration/ -x
```

Pre-commit hooks are configured in `.pre-commit-config.yaml` and enforce formatting/linting automatically. Always run `pre-commit run -a` before opening a PR.

## Git workflow

- **Default branch for PRs**: `develop` (not `main`)
- Fork → branch → commit → PR to `speechbrain/speechbrain:develop`
- CI runs on every PR: linting, doctests, unit tests, integration tests
- Each feature or fix gets its own branch

## Code style

- **Formatting and linting**: enforced by `ruff` via pre-commit. Run `pre-commit run -a`.
- **Docstrings**: NumPy-style. Every public class/function needs a docstring with description, arguments, returns, and a runnable example. Doctests are enforced by CI.
- **Self-documenting code**: prefer clear naming over comments. Use comments for surprising implementations or algorithm clarifications.
- **Minimize dependencies**: core library should stay lightweight. Recipe-specific deps go in `extra-requirements.txt`. If adding to core but the dependency is heavy, put it in `speechbrain/integrations/`.
- **Efficiency**: operate on batches, prefer tensor ops over Python loops, avoid CPU/GPU syncs.

## Common pitfalls

- **HyperPyYAML is not plain YAML**: do not treat `.yaml` files as simple config. `!new:` instantiates objects, `!ref` resolves references. Editing these files requires understanding the tag system. If you break a `!ref` chain, training will crash at load time.
- **Relative lengths, not absolute**: SpeechBrain passes relative lengths (0 to 1) for masking/padding. Do not pass absolute sample counts where relative lengths are expected.
- **modules vs hparams**: objects listed under `modules:` in the YAML are registered as `nn.Module`s on the Brain (moved to device, included in DDP, saved in checkpoints). Objects accessed via `self.hparams.*` are not. Putting a trainable module only in hparams means it won't be on the right device or saved properly.
- **Stage-dependent logic**: always check `stage` before computing validation-only metrics or applying train-only augmentation. Forgetting this causes training-time metric computation (slow) or test-time augmentation (wrong results).
- **Batch format**: batch objects from the dataio pipeline are `PaddedBatch` instances. Access signals as `batch.sig` which returns `(tensor, lengths)` tuples. Do not index batch like a plain dict.
- **Checkpointing**: Brain's checkpointer saves/loads modules, optimizers, schedulers, and epoch counters. If you add a new trainable module, register it with the checkpointer or it won't be saved/restored.
- **Soundfile vs torchaudio**: SpeechBrain is migrating audio I/O from torchaudio to soundfile. Use `speechbrain.dataio.dataio.read_audio` for reading audio, not raw torchaudio calls.

## Pretrained models and inference

SpeechBrain hosts pretrained models on HuggingFace at `huggingface.co/speechbrain/`. Inference interfaces in `speechbrain.inference` provide simple APIs:

```python
from speechbrain.inference.ASR import EncoderDecoderASR
asr = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech")
transcription = asr.transcribe_file("audio.wav")
```

The `from_hparams` method downloads the model and YAML from HuggingFace, loads via HyperPyYAML, and returns a ready-to-use object.

## Key dependencies

- PyTorch (core)
- HyperPyYAML (`hyperpyyaml` package — SpeechBrain's extended YAML, separate repo at `speechbrain/HyperPyYAML`)
- soundfile (audio I/O)
- torchaudio (some legacy audio I/O, being phased out)
- HuggingFace Hub (model hosting/downloading)
- sentencepiece / tokenizers (for text tokenization in some recipes)

## Links

- Documentation: https://speechbrain.readthedocs.io
- HyperPyYAML tutorial: https://speechbrain.readthedocs.io/en/latest/tutorials/basics/hyperpyyaml.html
- Contributing guide: https://speechbrain.readthedocs.io/en/latest/contributing.html
- Pretrained models: https://huggingface.co/speechbrain
- Benchmarks: https://github.com/speechbrain/benchmarks
