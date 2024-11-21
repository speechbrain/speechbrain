#!/usr/bin/env/python3
"""Recipe for training the LTU-AS model that jointly understands audio and speech.
This code is reformulated and enhanced based on the following github project:
https://github.com/YuanGongND/ltu

Authors
* Yingzhi Wang 2024
"""

import logging
import sys

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.data_utils import download_file

logger = logging.getLogger(__name__)

PRETRAINED_TLTR_URL = "https://www.dropbox.com/scl/fi/nciysewp1cedc3ob8etqe/large-v1_ori.pth?rlkey=2ekg4x0wpqlzxt4it92kqas7c&st=l25hte4d&dl=1"
# The 3 following urls are used for recipe tests
PRETRAINED_MODEL_STAGE1_URL = "https://www.dropbox.com/scl/fi/9m7e8z5luec8oyni5ixsj/model.ckpt?rlkey=xspvyql0dvotp15xkda1pemyh&st=5yzjaku1&dl=1"
PRETRAINED_MODEL_STAGE2_URL = "https://www.dropbox.com/scl/fi/nzv1ee724r1utno1rt2jy/model.ckpt?rlkey=ogaj5sfwj7i24o2jvxdfdexg4&st=vo0wx3jt&dl=1"
PRETRAINED_LLAMA_STAGE2_URL = "https://www.dropbox.com/scl/fi/4j69rgo6o1b6yk5wncwpf/llama3.ckpt?rlkey=ey304kf5ng5vgfq6ikh1cd15m&st=a2bj9npw&dl=1"


class ASLLMBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        audio_embs, _ = batch.audio_embs
        input_ids, _ = batch.input_ids

        # compute audio embedding
        audio_embs = self.modules.tltr(audio_embs)  # [b, 25, 1280]
        audio_embs = self.modules.audio_proj(audio_embs)  # [b, 25, 4096]

        # compute text embedding and concatenate
        input_embed = self.embedding_layer(input_ids)
        input_embed = torch.concat([audio_embs, input_embed], dim=1)

        # compute padding mask for audio and text, then concatenate
        text_padding_mask = ~self.hparams.text_padding_mask(
            input_ids, pad_idx=0
        )
        text_padding_mask = text_padding_mask.long()
        audio_padding_mask = torch.ones(
            [text_padding_mask.shape[0], 25], device=self.device
        )
        input_mask = torch.concat(
            [audio_padding_mask, text_padding_mask], dim=1
        )

        # forward llama
        outputs = self.modules.llama3(
            inputs_embeds=input_embed,
            attention_mask=input_mask,
        ).logits[:, 25:, :]

        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the NLL-loss using reply as label."""
        # Get required data from batch
        batch = batch.to(self.device)
        lm_labels, _ = batch.lm_labels
        audio_embs, _ = batch.audio_embs

        loss = self.hparams.ce_loss(
            predictions.flatten(end_dim=-2), lm_labels.flatten()
        )

        prompt_bos, _ = batch.user_prompt_bos
        reply_eos, reply_lens = batch.res_eos

        if stage == sb.Stage.VALID:
            batch = batch.to(self.device)
            audio_embs = self.modules.tltr(audio_embs)
            audio_embs = self.modules.audio_proj(audio_embs)
            input_embed = self.embedding_layer(prompt_bos)
            input_embed = torch.concat([audio_embs, input_embed], dim=1)
            text_padding_mask = ~self.hparams.text_padding_mask(
                prompt_bos, pad_idx=0
            )
            text_padding_mask = text_padding_mask.long()
            audio_padding_mask = torch.ones(
                [text_padding_mask.shape[0], 25], device=self.device
            )
            input_mask = torch.concat(
                [audio_padding_mask, text_padding_mask], dim=1
            )

            if (
                hasattr(self.hparams, "stage2_llama_path")
                and not self.hparams.recipe_test
            ):
                # stage3
                hyps = self.modules.llama3.module.generate(
                    inputs_embeds=input_embed.detach(),
                    attention_mask=input_mask.detach(),
                )
            else:
                # stage1 & 2
                hyps = self.modules.llama3.generate(
                    inputs_embeds=input_embed.detach(),
                    attention_mask=input_mask.detach(),
                )

            predicted_words = tokenizer.batch_decode(
                hyps,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            target_words = tokenizer.batch_decode(
                reply_eos,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            if stage != sb.Stage.TRAIN:
                self.hyps.extend(predicted_words)
                self.references.extend(target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.hyps = []
            self.references = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing.current_lr
            steps = self.optimizer_step
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": steps,
                    "lr": lr,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]},
                min_keys=["loss"],
            )
            if epoch == hparams["number_of_epochs"] - 1:
                with open(self.hparams.valid_file, "w", encoding="utf-8") as w:
                    for i in range(len(self.hyps)):
                        w.write("target: " + str(self.references[i]) + "\n")
                        w.write("predicted:" + str(self.hyps[i]) + "\n")
                        w.write(
                            "+++++++++++++++++++++++++++++++++++++++++++++++\n"
                        )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing and logging."""
        if should_step:
            self.hparams.lr_annealing(self.optimizer)

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def tokenize(tokenizer, prompt, cutoff_len):
    result = tokenizer.encode(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    return result


def dataio_prep(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_openasqa` to have been called before
    this, so that the `train.json` and `test.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    tokenizer : tokenizer
        Object for converting strings to tokens.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" (if needed) that correspond
        to the appropriate DynamicItemDataset object.
    """

    @sb.utils.data_pipeline.takes("feature_path")
    @sb.utils.data_pipeline.provides("audio_embs")
    def audio_pipeline(feature_path):
        audio_emb = np.load(feature_path)["arr_0"]
        return torch.from_numpy(audio_emb)

    @sb.utils.data_pipeline.takes("instruction", "input", "output")
    @sb.utils.data_pipeline.provides(
        "user_prompt_bos", "res_eos", "input_ids", "lm_labels"
    )
    def text_pipeline_llama3(instruction, input, output):
        # the llama3 template
        user_prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an assistant that understands audio and speech.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction} The transcript of the audio is:{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        user_prompt_tokens = tokenize(
            tokenizer, user_prompt, hparams["cutoff_len"]
        )[1:]
        user_prompt_bos = torch.cat(
            (
                torch.tensor([tokenizer.bos_token_id]),
                torch.LongTensor(user_prompt_tokens),
            )
        )
        yield user_prompt_bos

        res_tokens = tokenize(tokenizer, output, hparams["cutoff_len"])[1:]
        res_eos = torch.cat(
            (
                torch.LongTensor(res_tokens),
                torch.tensor([tokenizer.eos_token_id]),
            )
        )
        yield res_eos

        input_ids = torch.cat(
            (
                torch.tensor([tokenizer.bos_token_id]),
                torch.LongTensor(user_prompt_tokens + res_tokens),
            )
        )
        yield input_ids

        if not hparams["train_on_input"]:
            user_prompt_len = len(user_prompt_tokens)
            labels = torch.cat(
                (
                    torch.LongTensor([-100] * user_prompt_len),
                    res_eos,
                )
            )
        else:
            labels = torch.cat(
                (
                    torch.LongTensor(user_prompt_tokens),
                    res_eos,
                )
            )
        yield labels

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[
                audio_pipeline,
                text_pipeline_llama3,
            ],
            output_keys=[
                "id",
                "audio_embs",
                "user_prompt_bos",
                "res_eos",
                "input_ids",
                "lm_labels",
            ],
        )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load tokenizer and add special tokens
    tokenizer = hparams["llama3"].tokenizer
    # tokenizer.add_bos_token = False # does not work for llama3 tokenizer

    #  Load pretrained LLAMA3
    hparams["llama3"] = hparams["llama3"].to(device=run_opts["device"])
    hparams["tltr"] = hparams["tltr"].to(device=run_opts["device"])

    class CustomPaddedBatch(PaddedBatch):
        """PaddedBatch with custom padding values.

        See the documentation of `speechbrain.dataio.batch.PaddedBatch`.

        """

        def __init__(self, examples, *args, **kwargs):
            for k in [
                "input_ids",
                "user_prompt_bos",
                "lm_labels",
            ]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0
                if k == "lm_labels":
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    if k in ["user_prompt_bos"]:
                        example[k] = torch.nn.functional.pad(
                            x, [max_len - len(x), 0], value=pad_value
                        )
                    else:
                        example[k] = torch.nn.functional.pad(
                            x, [0, max_len - len(x)], value=pad_value
                        )
            super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_opts"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_opts"]["collate_fn"] = CustomPaddedBatch

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams, tokenizer)

    # Initialize the Brain object to prepare for mask training.
    asllm_brain = ASLLMBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_models" in hparams:
        # load pretrained models from previous stage for stage 2 and 3
        if hparams["recipe_test"]:
            # need to download the pretrained models if this is a recipe test
            if "stage1_model_path" in hparams:
                # stage 2
                download_file(
                    PRETRAINED_MODEL_STAGE1_URL, hparams["stage1_model_path"]
                )
            else:
                # stage 3
                download_file(
                    PRETRAINED_MODEL_STAGE2_URL, hparams["stage2_model_path"]
                )
                download_file(
                    PRETRAINED_LLAMA_STAGE2_URL, hparams["stage2_llama_path"]
                )

        sb.utils.distributed.run_on_main(
            hparams["pretrained_models"].collect_files
        )
        hparams["pretrained_models"].load_collected()
        logger.info(
            "Pretrained models from previous stage loaded, this stage should not be stage1"
        )
    else:
        # download the tltr weights for stage 1
        download_file(PRETRAINED_TLTR_URL, hparams["tltr_pretrained_weights"])
        hparams["tltr"].load_state_dict(
            torch.load(hparams["tltr_pretrained_weights"]),
            strict=True,
        )
        logger.info(
            "TLTR loaded with pretrained weights, this stage should be stage1"
        )

    asllm_brain.embedding_layer = hparams["llama3"].model.get_input_embeddings()

    asllm_brain.fit(
        epoch_counter=asllm_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
