# PyTest for reporting code coverage rates

How to know test coverage changes of Open PRs to be merged?
<br/>_(snippet for cpu-only)_
```
# Example: install more dependencies to avoid ignoring modules
sudo apt-get install -y libsndfile1
pip install ctc_segmentation

# install coverage
pip install pytest-cov

# run the test (w/ duration reporting)
pytest --durations=0 --cov=speechbrain --cov-context=test --doctest-modules speechbrain tests --ignore=speechbrain/nnet/loss/transducer_loss.py
```
Example: _After collecting 459 testing items, 4481/16782 statements are reported "missing" (73% coverage)._

YET—python code of the core modules is not all to be covered; thus far, only, consistency is ensured for III.A (through III.B).

---

Further reading:
<br/> pytest & coverage - https://breadcrumbscollector.tech/how-to-use-code-coverage-in-python-with-pytest/ (pointer by @Adel-Moumen)

---

```
pytest --durations=0 --cov=speechbrain --cov-context=test --doctest-modules speechbrain tests --ignore=speechbrain/nnet/loss/transducer_loss.py

---------- coverage: platform linux, python 3.9.12-final-0 -----------
Name                                                      Stmts   Miss  Cover
-----------------------------------------------------------------------------
speechbrain/alignment/aligner.py                            380     61    84%
speechbrain/alignment/ctc_segmentation.py                   189     10    95%
speechbrain/core.py                                         424    155    63% <== < 80%
speechbrain/dataio/batch.py                                  99      8    92%
speechbrain/dataio/dataio.py                                279     50    82%
speechbrain/dataio/dataloader.py                            140     25    82%
speechbrain/dataio/dataset.py                               100      8    92%
speechbrain/dataio/encoder.py                               328     46    86%
speechbrain/dataio/iterators.py                              80     62    22% <== < 80%
speechbrain/dataio/legacy.py                                121     41    66% <== < 80%
speechbrain/dataio/preprocess.py                             22      4    82%
speechbrain/dataio/sampler.py                               224     61    73% <== < 80%
speechbrain/dataio/wer.py                                    63     54    14% <== < 80%
speechbrain/decoders/ctc.py                                 111     89    20% <== < 80%
speechbrain/decoders/seq2seq.py                             370     46    88%
speechbrain/decoders/transducer.py                          133     64    52% <== < 80%
speechbrain/lm/arpa.py                                       77      3    96%
speechbrain/lm/counting.py                                   37      4    89%
speechbrain/lm/ngram.py                                      36      1    97%
speechbrain/lobes/augment.py                                154     55    64% <== < 80%
speechbrain/lobes/beamform_multimic.py                       20     14    30% <== < 80%
speechbrain/lobes/features.py                                96      9    91%
speechbrain/lobes/models/CRDNN.py                            52     12    77% <== < 80%
speechbrain/lobes/models/ContextNet.py                       83      3    96%
speechbrain/lobes/models/ECAPA_TDNN.py                      157      7    96%
speechbrain/lobes/models/HifiGAN.py                         321    146    55% <== < 80%
speechbrain/lobes/models/MetricGAN.py                        74     29    61% <== < 80%
speechbrain/lobes/models/Tacotron2.py                       364     66    82%
speechbrain/lobes/models/conv_tasnet.py                     121      6    95%
speechbrain/lobes/models/dual_path.py                       357     55    85%
speechbrain/lobes/models/fairseq_wav2vec.py                  93     93     0% <== < 80%
speechbrain/lobes/models/g2p/dataio.py                      136    107    21% <== < 80%
speechbrain/lobes/models/g2p/homograph.py                   118     20    83%
speechbrain/lobes/models/g2p/model.py                       132    109    17% <== < 80%
speechbrain/lobes/models/huggingface_wav2vec.py             145     47    68% <== < 80%
speechbrain/lobes/models/resepformer.py                     180     21    88%
speechbrain/lobes/models/segan_model.py                     102     88    14% <== < 80%
speechbrain/lobes/models/transformer/Conformer.py           111      7    94%
speechbrain/lobes/models/transformer/Transformer.py         180     22    88%
speechbrain/lobes/models/transformer/TransformerASR.py       92     28    70% <== < 80%
speechbrain/lobes/models/transformer/TransformerLM.py        47      5    89%
speechbrain/lobes/models/transformer/TransformerSE.py        20      2    90%
speechbrain/lobes/models/transformer/TransformerST.py        81     60    26% <== < 80%
speechbrain/lobes/models/wav2vec.py                         123     55    55% <== < 80%
speechbrain/nnet/CNN.py                                     417     56    87%
speechbrain/nnet/RNN.py                                     471     51    89%
speechbrain/nnet/activations.py                              39      1    97%
speechbrain/nnet/attention.py                               234     44    81%
speechbrain/nnet/complex_networks/c_CNN.py                  130     23    82%
speechbrain/nnet/complex_networks/c_RNN.py                  374     67    82%
speechbrain/nnet/complex_networks/c_normalization.py        277     68    75% <== < 80%
speechbrain/nnet/complex_networks/c_ops.py                  108     40    63% <== < 80%
speechbrain/nnet/containers.py                              139     14    90%
speechbrain/nnet/linear.py                                   27      1    96%
speechbrain/nnet/loss/si_snr_loss.py                         20     16    20% <== < 80%
speechbrain/nnet/loss/stoi_loss.py                           81      1    99%
speechbrain/nnet/loss/transducer_loss.py                    136    136     0% <== < 80%
speechbrain/nnet/losses.py                                  323    112    65% <== < 80%
speechbrain/nnet/normalization.py                           142      6    96%
speechbrain/nnet/pooling.py                                 156     31    80%
speechbrain/nnet/quantisers.py                               47      2    96%
speechbrain/nnet/quaternion_networks/q_CNN.py               150     25    83%
speechbrain/nnet/quaternion_networks/q_RNN.py               370     59    84%
speechbrain/nnet/quaternion_networks/q_linear.py             50     11    78% <== < 80%
speechbrain/nnet/quaternion_networks/q_normalization.py      44      4    91%
speechbrain/nnet/quaternion_networks/q_ops.py               229    122    47% <== < 80%
speechbrain/nnet/schedulers.py                              363    103    72% <== < 80%
speechbrain/nnet/transducer/transducer_joint.py              33      5    85%
speechbrain/pretrained/fetching.py                           48      6    88%
speechbrain/pretrained/interfaces.py                        786    338    57% <== < 80%
speechbrain/pretrained/training.py                           33     28    15% <== < 80%
speechbrain/processing/PLDA_LDA.py                          345     96    72% <== < 80%
speechbrain/processing/decomposition.py                     102      8    92%
speechbrain/processing/diarization.py                       319    157    51% <== < 80%
speechbrain/processing/features.py                          359     75    79% <== < 80%
speechbrain/processing/multi_mic.py                         345      2    99%
speechbrain/processing/signal_processing.py                 166     39    77% <== < 80%
speechbrain/processing/speech_augmentation.py               386     34    91%
speechbrain/tokenizers/SentencePiece.py                     181     74    59% <== < 80%
speechbrain/utils/Accuracy.py                                24     17    29% <== < 80%
speechbrain/utils/DER.py                                     44     33    25% <== < 80%
speechbrain/utils/bleu.py                                    50     43    14% <== < 80%
speechbrain/utils/callchains.py                              28      5    82%
speechbrain/utils/checkpoints.py                            294     52    82%
speechbrain/utils/data_pipeline.py                          181     15    92%
speechbrain/utils/data_utils.py                             197     77    61% <== < 80%
speechbrain/utils/depgraph.py                                82      1    99%
speechbrain/utils/distributed.py                             61     37    39% <== < 80%
speechbrain/utils/edit_distance.py                          180     50    72% <== < 80%
speechbrain/utils/epoch_loop.py                              55     22    60% <== < 80%
speechbrain/utils/hparams.py                                  2      1    50% <== < 80%
speechbrain/utils/hpopt.py                                  134     41    69% <== < 80%
speechbrain/utils/logger.py                                  73     45    38% <== < 80%
speechbrain/utils/metric_stats.py                           285     48    83%
speechbrain/utils/parameter_transfer.py                      87     17    80%
speechbrain/utils/profiling.py                              191     54    72% <== < 80%
speechbrain/utils/superpowers.py                             20      6    70% <== < 80%
speechbrain/utils/text_to_sequence.py                        77     22    71% <== < 80%
speechbrain/utils/torch_audio_backend.py                      9      2    78% <== < 80%
speechbrain/utils/train_logger.py                           150    113    25% <== < 80%
speechbrain/wordemb/transformer.py                           90     67    26% <== < 80%
-----------------------------------------------------------------------------
TOTAL                                                     16782   4481    73%
```