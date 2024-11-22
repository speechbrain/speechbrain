"""Recipe for evaluating a grapheme-to-phoneme system with librispeech lexicon.

The script may be use in isolation or in combination with Orion to fit
hyperparameters that do not require model retraining (e.g. Beam Search)

Authors
 * Mirco Ravanelli 2022
 * Artem Ploujnikov 2022
"""

import itertools
import math
import sys
from types import SimpleNamespace

import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm.auto import tqdm
from train import dataio_prep, load_dependencies

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.models.g2p.dataio import get_sequence_key
from speechbrain.utils import hpopt as hp
from speechbrain.utils.logger import get_logger
from speechbrain.wordemb.util import expand_to_chars

logger = get_logger(__name__)


class G2PEvaluator:
    """The G2P model evaluation wrapper

    Arguments
    ---------
    hparams: dict
        the dictionary from a parsed hyperparameter file
    device: str
        the device identifier
    model_state: dict
        a pre-loaded model state for a "warm start" if applicable
        - could be useful if hyperparameters have changed, but
        the same model can be reused from one run to the next
    """

    def __init__(self, hparams, device, model_state=None):
        self.hparams = SimpleNamespace(**hparams)
        self.overrides = overrides
        self.device = device
        self.modules = torch.nn.ModuleDict(self.hparams.modules).to(self.device)
        beam_searcher = (
            self.hparams.beam_searcher_lm
            if self.hparams.use_language_model
            else self.hparams.beam_searcher
        )
        self.beam_searcher = beam_searcher.to(self.device)
        if model_state:
            self.hparams.model.load_state_dict(model_state)
        else:
            self.load()
        self.grapheme_sequence_mode = getattr(
            self.hparams, "grapheme_sequence_mode", "bos"
        )
        self.grapheme_key = get_sequence_key(
            key="grapheme_encoded", mode=self.grapheme_sequence_mode
        )
        self.modules["model"].eval()
        self._word_separator = None
        self._bos = torch.tensor(
            self.hparams.bos_index, device=device
        ).unsqueeze(-1)
        self._eos = torch.tensor(
            self.hparams.eos_index, device=device
        ).unsqueeze(-1)

        # When reconstructing sentences word-wise, the process depends
        # on whether spaces are preserved or omitted, as controlled by
        # the phonemes_enable_space hyperparameter
        self._flatten_results = (
            self._flatten_results_separated
            if getattr(self.hparams, "phonemes_enable_space", None)
            else self._flatten_results_jumbled
        )
        self._grapheme_word_separator_idx = None
        if self.hparams.use_word_emb:
            self.modules.word_emb = self.hparams.word_emb().to(self.device)

    def load(self):
        """Loads a model from a checkpoint"""
        checkpointer = self.hparams.checkpointer
        ckpt = checkpointer.recover_if_possible(
            device=torch.device(self.device),
            importance_key=lambda ckpt: -ckpt.meta.get("PER", -100.0),
            ckpt_predicate=lambda ckpt: ckpt.meta["step"]
            == self.hparams.eval_ckpt_step,
        )
        if ckpt:
            logger.info("Loaded checkpoint with metadata %s", ckpt.meta)
        else:
            raise ValueError(
                f"Checkpoint not found for training step {self.hparams.eval_train_step}"
            )
        return ckpt

    def evaluate_batch(self, batch):
        """
        Evaluates the G2P model

        Arguments
        ---------
        batch: PaddedBatch
            A single batch of data, same as the kind of batch used
            for G2P training
        """
        batch = batch.to(self.device)
        grapheme_encoded = getattr(batch, self.grapheme_key)
        if self.hparams.eval_mode == "sentence":
            hyps, scores = self._get_phonemes(grapheme_encoded, char=batch.char)
        elif self.hparams.eval_mode == "word":
            hyps, scores = self._get_phonemes_wordwise(batch.grapheme_encoded)
        else:
            raise ValueError(f"unsupported eval_mode {self.hparams.eval_mode}")

        ids = batch.sample_id

        phns, phn_lens = batch.phn_encoded

        self.per_metrics.append(
            ids, hyps, phns, None, phn_lens, self.hparams.out_phoneme_decoder
        )

    def _get_phonemes(self, grapheme_encoded, phn_encoded=None, char=None):
        """Runs the model and the beam search to retrieve the phoneme sequence
        corresponding to the provided grapheme sequence

        Arguments
        ---------
        grapheme_encoded: speechbrain.dataio.batch.PaddedData
            An encoded grapheme sequence
        phn_encoded: speechbrain.dataio.batch.PaddedData
            An encoded phoneme sequence (optional)
        char: str
            Raw character input (needed for word embeddings)

        Returns
        -------
        hyps: list
            the hypotheses (the beam search result)
        scores: list
            the scores corresponding to the hypotheses
        """
        _, char_word_emb = None, None
        if self._grapheme_word_separator_idx is None:
            self._grapheme_word_separator_idx = (
                self.hparams.grapheme_encoder.lab2ind[" "]
            )
        if not phn_encoded:
            grapheme_encoded_data, grapheme_lens = grapheme_encoded
            phn_encoded = (
                torch.ones(len(grapheme_encoded_data), 1).to(
                    grapheme_encoded_data.device
                )
                * self.hparams.bos_index,
                torch.ones(len(grapheme_encoded_data)).to(
                    grapheme_encoded_data.device
                ),
            )
            char_word_emb = self._apply_word_embeddings(grapheme_encoded, char)
        p_seq, char_lens, encoder_out, _ = self.modules.model(
            grapheme_encoded=grapheme_encoded,
            phn_encoded=phn_encoded,
            word_emb=char_word_emb,
        )
        return self.beam_searcher(encoder_out, char_lens)

    def _apply_word_embeddings(self, grapheme_encoded, char):
        char_word_emb = None
        if self.hparams.use_word_emb:
            grapheme_encoded_data, grapheme_lens = grapheme_encoded
            word_emb = self.modules.word_emb.batch_embeddings(char)
            char_word_emb = expand_to_chars(
                emb=word_emb,
                seq=grapheme_encoded_data,
                seq_len=grapheme_lens,
                word_separator=self._grapheme_word_separator_idx,
            )
        return char_word_emb

    def _get_phonemes_wordwise(self, grapheme_encoded):
        """Retrieves the phoneme sequence corresponding to the provided grapheme
        sequence in a word-wise manner (running the evaluator for each word separately)

        Arguments
        ---------
        grapheme_encoded: speechbrain.dataio.batch.PaddedData
            An encoded grapheme sequence

        Returns
        -------
        hyps: list
            the hypotheses (the beam search result)
        scores: list
            the scores corresponding to the hypotheses
        """
        if self.hparams.use_word_emb:
            raise NotImplementedError(
                "Wordwise evaluation is not supported with word embeddings"
            )
        if self._word_separator is None:
            self._word_separator = self.hparams.phoneme_encoder.lab2ind[" "]
        hyps, scores = [], []
        for grapheme_item, grapheme_len in zip(
            grapheme_encoded.data, grapheme_encoded.lengths
        ):
            words_batch = self._split_words_batch(grapheme_item, grapheme_len)
            item_hyps, item_scores = self._get_phonemes(
                words_batch.grapheme_encoded
            )
            hyps.append(self._flatten_results(item_hyps))
            scores.append(self._flatten_scores(item_hyps, item_scores))
        return hyps, scores

    def _flatten_results_jumbled(self, results):
        """Flattens a sequence of results into a single sequence of tokens -
        used when spaces are preserved in the phoneme space

        Arguments
        ---------
        results: iterable
            a two-dimensional result

        Returns
        -------
        result: list
            the concatenated result
        """
        return [token for item_result in results for token in item_result]

    def _flatten_results_separated(self, results):
        """Flattens a sequence of words, inserting word separators between them -
        used when word separators are preserved in the phoneme space

        Arguments
        ---------
        results: iterable
            a two-dimensional result

        Returns
        -------
        result: list
            the concatenated result
        """
        result = []
        for item_result in results:
            for token in item_result:
                result.append(token)
            if item_result and item_result[-1] != self._word_separator:
                result.append(self._word_separator)
        del result[-1]
        return result

    def _flatten_scores(self, hyps, scores):
        """Flattens an array of scores, using a weighted average of the scores of
        individual words, by word length

        Arguments
        ---------
        hyps: list
            the hypotheses (the beam search result)
        scores: list
            the scores corresponding to the hypotheses

        Returns
        -------
        scores: list
            the scores corresponding to the hypotheses,
            merged
        """
        seq_len = sum(len(word_hyp) for word_hyp in hyps)
        return (
            sum(
                word_score * len(word_hyp)
                for word_hyp, word_score in zip(hyps, scores)
            )
            / seq_len
        )

    def _split_words_batch(self, graphemes, length):
        return PaddedBatch(
            [
                {"grapheme_encoded": word}
                for word in self._split_words_seq(graphemes, length)
            ]
        ).to(self.device)

    def _split_words_seq(self, graphemes, length):
        """Splits the provided grapheme sequence into words

        Arguments
        ---------
        graphemes: torch.Tensor
            an encoded sequence of phonemes
        length: torch.Tensor
            The length of the corresponding inputs.

        Yields
        ------
        graphemes: generator
            a generator representing a sequence of words
        """
        space_index = self.hparams.graphemes.index(" ")
        (word_boundaries,) = torch.where(graphemes == space_index)
        last_word_boundary = 0
        for word_boundary in word_boundaries:
            yield self._add_delimiters(
                graphemes[last_word_boundary + 1 : word_boundary]
            )
            last_word_boundary = word_boundary
        char_length = math.ceil(len(graphemes) * length)
        if last_word_boundary < char_length:
            yield self._add_delimiters(
                graphemes[last_word_boundary + 1 : char_length]
            )

    def _add_delimiters(self, word):
        """Adds the required delimiter characters to a word

        Arguments
        ---------
        word: torch.Tensor
            a tensor representing a word

        Returns
        -------
        word: torch.Tensor
            word with delimiters added.
        """
        if self.grapheme_sequence_mode == "bos":
            word = torch.cat([self._bos, word])
        elif self.grapheme_sequence_mode == "eos":
            word = torch.cat([word, self._eos])
        return word

    def evaluate_epoch(self, dataset, dataloader_opts=None):
        """
        Evaluates a single epoch

        Arguments
        ---------
        dataset: DynamicItemDataset
            a G2P dataset (same as the ones used for training)
        dataloader_opts: dict
            Additional options to pass to dataloader.

        Returns
        -------
        metrics: dict
            Raw PER metrics
        """
        logger.info("Beginning evaluation")
        with torch.no_grad():
            self.per_metrics = self.hparams.per_stats()
            dataloader = sb.dataio.dataloader.make_dataloader(
                dataset,
                **dict(
                    dataloader_opts or {},
                    shuffle=True,
                    batch_size=self.hparams.eval_batch_size,
                ),
            )
            dataloader_it = iter(dataloader)
            if self.hparams.eval_batch_count is not None:
                dataloader_it = itertools.islice(
                    dataloader_it, 0, self.hparams.eval_batch_count
                )
                batch_count = self.hparams.eval_batch_count
            else:
                batch_count = math.ceil(
                    len(dataset) / self.hparams.eval_batch_size
                )
            for batch in tqdm(dataloader_it, total=batch_count):
                self.evaluate_batch(batch)
            if self.hparams.eval_output_wer_file:
                self._output_wer_file()
            return self.per_metrics.summarize()

    def _output_wer_file(self):
        with open(self.hparams.eval_wer_file, "w", encoding="utf-8") as w:
            w.write("\nPER stats:\n")
            self.per_metrics.write_stats(w)
            print(
                "seq2seq, and PER stats written to file",
                self.hparams.eval_wer_file,
            )


if __name__ == "__main__":
    # CLI:

    with hp.hyperparameter_optimization(objective_key="error_rate") as hp_ctx:
        # Parse the hyperparameter file
        search_hparam_file = sys.argv[0]
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:])
        device = run_opts.get("device", "cpu")
        with open(hparams_file, encoding="utf-8") as fin:
            hparams = load_hyperpyyaml(fin, overrides)

        # Load dependencies
        if hparams.get("use_language_model"):
            load_dependencies(hparams, run_opts)

        # Run the evaluation
        evaluator = G2PEvaluator(hparams, device)

        # Some configurations involve curriculum training on
        # multiple steps. Load the dataset configuration for the
        # step specified in the eval_train_step hyperparameter
        # (or command-line argument)
        train_step = next(
            train_step
            for train_step in hparams["train_steps"]
            if train_step["name"] == hparams["eval_train_step"]
        )
        train, valid, test, _ = dataio_prep(hparams, train_step)
        datasets = {"train": train, "valid": valid, "test": test}
        dataset = datasets[hparams["eval_dataset"]]
        dataloader_opts = train_step.get(
            "dataloader_opts", hparams.get("dataloader_opts", {})
        )
        result = evaluator.evaluate_epoch(dataset, dataloader_opts)
        hp.report_result(result)
