#!/usr/bin/env/python3
"""Example recipes to benchmark SpeechBrain using PyTorch profiling; print and export tensorboard & FlameGraph reports.

This file demonstrates how to use the @profile_optimiser decorator for:
    1. speechbrain.Brain.fit()
    2. speechbrain.Brain.evaluate()
    3. when to use speechbrain.dataio.dataloader.make_dataloader(TOO_SMALL_DATASET, looped_nominal_epoch=SCHEDULER_FIN)

Showcase: tests/integration/neural_networks/ASR_seq2seq/example_asr_seq2seq_experiment.py
            // integration testing related code lines are removed.

Author:
    * Andreas Nautsch 2022
"""
import pathlib
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.profiling import profile_optimiser  # import added


@profile_optimiser  # <=== only added line of code (check below for where to print what) - run & check `log` folder :)
class seq2seqBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns_bos, _ = batch.phn_encoded_bos
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, wav_lens)
        x = self.modules.enc(feats)

        # Prepend bos token at the beginning
        e_in = self.modules.emb(phns_bos)
        h, w = self.modules.dec(e_in, x, wav_lens)
        logits = self.modules.lin(h)
        outputs = self.hparams.softmax(logits)

        if stage != sb.Stage.TRAIN:
            seq, _ = self.hparams.searcher(x, wav_lens)
            return outputs, seq

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the NLL loss."
        if stage == sb.Stage.TRAIN:
            outputs = predictions
        else:
            outputs, seq = predictions

        ids = batch.id
        phns, phn_lens = batch.phn_encoded_eos

        loss = self.hparams.compute_cost(outputs, phns, length=phn_lens)

        if stage != sb.Stage.TRAIN:
            self.per_metrics.append(ids, seq, phns, target_len=phn_lens)

        return loss

    def fit_batch(self, batch):
        """Fits train batches"""
        preds = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(preds, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage=sb.Stage.TEST):
        """Evaluates test batches"""
        out = self.compute_forward(batch, stage)
        loss = self.compute_objectives(out, batch, stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Gets called when a stage (either training, validation, test) ends."
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.VALID and epoch is not None:
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
        if stage != sb.Stage.TRAIN:
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % self.per_metrics.summarize("error_rate"))


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "train.json",
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / "dev.json",
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]
    label_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides(
        "phn_list", "phn_encoded_bos", "phn_encoded_eos"
    )
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        phn_encoded_bos = label_encoder.prepend_bos_index(phn_encoded).long()
        yield phn_encoded_bos
        phn_encoded_eos = label_encoder.append_eos_index(phn_encoded).long()
        yield phn_encoded_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # NOTE: In this minimal example, also update from valid data
    label_encoder.insert_bos_eos(bos_index=hparams["bos_index"])
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    label_encoder.update_from_didataset(valid_data, output_key="phn_list")

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "phn_encoded_eos", "phn_encoded_bos"]
    )
    return train_data, valid_data


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparams_file = "../../tests/integration/neural_networks/ASR_seq2seq/hyperparams.yaml"  # path adjusted
    data_folder = (
        "../../samples/audio_samples/nn_training_samples"  # path adjusted
    )
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    seq2seq_brain = seq2seqBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
    )

    # Training/validation loop
    seq2seq_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # <=== added lines to report benchmark
    """print(seq2seq_brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*         8.31%      66.600ms        99.92%     800.909ms     400.454ms     160.16 Kb     -28.72 Mb             2
                                                   aten::mm        13.66%     109.495ms        13.77%     110.389ms      95.907us      77.98 Mb      77.98 Mb          1151
                                                aten::copy_        11.02%      88.313ms        11.02%      88.313ms      52.380us           0 b           0 b          1686
    autograd::engine::evaluate_function: MkldnnConvoluti...         0.02%     158.000us        10.81%      86.655ms      21.664ms     -31.48 Mb     -46.98 Mb             4
                                 MkldnnConvolutionBackward0         0.00%      38.000us        10.79%      86.497ms      21.624ms      15.49 Mb           0 b             4
                          aten::mkldnn_convolution_backward         0.02%     134.000us        10.79%      86.459ms      21.615ms      15.49 Mb     -30.53 Mb             4
                                                aten::clone         0.16%       1.277ms         9.75%      78.118ms     398.561us     168.21 Mb           0 b           196
           autograd::engine::evaluate_function: MmBackward0         0.64%       5.098ms         9.55%      76.545ms     276.336us      37.97 Mb      -9.03 Mb           277
                                           aten::contiguous         0.02%     178.000us         9.33%      74.793ms       2.200ms     164.34 Mb           0 b            34
                                                MmBackward0         0.32%       2.561ms         8.68%      69.598ms     251.256us      46.84 Mb           0 b           277
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 801.557ms
    """
    # ===>

    # Evaluation is run separately (now just evaluating on valid data)
    seq2seq_brain.evaluate(valid_data)

    # <=== added lines to report benchmark
    """print(seq2seq_brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    // nothing — not enough samples in valid_data for the scheduler to warm-up ;-)
    ...
    results from fit() are stored in: seq2seq_brain.profiler.speechbrain_event_traces[0]
    // if evaluate() would have had processed enough batches (so scheduler starts recording), then there would be a:
                                      seq2seq_brain.profiler.speechbrain_event_traces[1] with its recorded benchmark
    ...
    Now, running evaluate() lots of times won't help - the scheduler always resets its counter to 0.
        seq2seq_brain.evaluate(valid_data)
        seq2seq_brain.evaluate(valid_data)
        seq2seq_brain.evaluate(valid_data)
        seq2seq_brain.evaluate(valid_data)
        seq2seq_brain.evaluate(valid_data)
        seq2seq_brain.evaluate(valid_data)
    ...
        => no impact, not worth trying.
    ...
    But, but - what to do instead?
    ...
        NOT - using @profile to avoid the scheduler obtains another measurement as @profile_optimiser (worthless report)
    ...
        N/A - reduce batch size, e.g.: seq2seq_brain.evaluate(valid_data, test_loader_kwargs={"batch_size": 1})
              Problem in this specific case: this will create two batches (two audio files in total)
    ...
        TRY - loop over dataset until scheduler recorded & finished    ! NOT for recognition performance reporting !
    ...
                seq2seq_brain.evaluate(sb.dataio.dataloader.make_dataloader(valid_data, looped_nominal_epoch=6))
    ...
    print(seq2seq_brain.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                              ProfilerStep*        18.48%      58.470ms        99.98%     316.355ms     158.178ms     174.08 Kb    -184.38 Mb             2
                                               aten::linear         1.14%       3.608ms        19.30%      61.062ms      83.078us       1.62 Mb           0 b           735
                                               aten::matmul         0.57%       1.805ms        13.12%      41.506ms      98.123us       1.39 Mb    -229.00 Kb           423
                                                aten::copy_        12.34%      39.032ms        12.34%      39.032ms      47.658us           0 b           0 b           819
                                                aten::clone         0.18%     581.000us        12.21%      38.622ms     382.396us      88.41 Mb           0 b           101
                                                   aten::mm        12.01%      38.009ms        12.13%      38.388ms      90.752us       1.39 Mb       1.39 Mb           423
                                           aten::contiguous         0.03%     104.000us        11.43%      36.156ms       1.808ms      83.76 Mb           0 b            20
                                           aten::layer_norm         0.02%      78.000us        10.34%      32.713ms       8.178ms      37.06 Mb      -7.19 Kb             4
                                    aten::native_layer_norm         3.95%      12.488ms        10.31%      32.635ms       8.159ms      37.07 Mb     -37.06 Mb             4
                                          aten::convolution         0.02%      74.000us         9.44%      29.854ms       2.985ms      38.29 Mb           0 b            10
    -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    Self CPU time total: 316.411ms
    ...
    print(len(seq2seq_brain.profiler.speechbrain_event_traces))  # 2 --> it's also stored; check the `log` folder :)
    """
    # ===>


if __name__ == "__main__":
    main()
