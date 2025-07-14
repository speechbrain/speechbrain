"""Specifies the inference interfaces for text-processing modules.

Authors:
 * Aku Rouhe 2021
 * Peter Plantinga 2021
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
 * Titouan Parcollet 2021
 * Abdel Heba 2021
 * Andreas Nautsch 2022, 2023
 * Pooneh Mousavi 2023
 * Sylvain de Langen 2023
 * Adel Moumen 2023
 * Pradnya Kandarkar 2023
"""

from itertools import chain

import torch

from speechbrain.inference.interfaces import (
    EncodeDecodePipelineMixin,
    Pretrained,
)


class GraphemeToPhoneme(Pretrained, EncodeDecodePipelineMixin):
    """
    A pretrained model implementation for Grapheme-to-Phoneme (G2P) models
    that take raw natural language text as an input and

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> text = (
    ...     "English is tough. It can be understood "
    ...     "through thorough thought though"
    ... )
    >>> from speechbrain.inference.text import GraphemeToPhoneme
    >>> tmpdir = getfixture("tmpdir")
    >>> g2p = GraphemeToPhoneme.from_hparams(
    ...     "path/to/model", savedir=tmpdir
    ... )  # doctest: +SKIP
    >>> phonemes = g2p.g2p(text)  # doctest: +SKIP
    """

    INPUT_STATIC_KEYS = ["txt"]
    OUTPUT_KEYS = ["phonemes"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_pipelines()
        self.load_dependencies()

    @property
    def phonemes(self):
        """Returns the available phonemes"""
        return self.hparams.phonemes

    @property
    def language(self):
        """Returns the language for which this model is available"""
        return self.hparams.language

    def g2p(self, text):
        """Performs the Grapheme-to-Phoneme conversion

        Arguments
        ---------
        text: str or list[str]
            a single string to be encoded to phonemes - or a
            sequence of strings

        Returns
        -------
        result: list
            if a single example was provided, the return value is a
            single list of phonemes
        """
        single = isinstance(text, str)
        if single:
            text = [text]

        encoded_inputs = self.encode_input({"txt": text})
        self._update_graphemes(encoded_inputs)

        model_inputs = encoded_inputs
        if hasattr(self.hparams, "model_input_keys"):
            model_inputs = {
                k: model_inputs[k] for k in self.hparams.model_input_keys
            }

        model_outputs = self.mods.model(**model_inputs)
        decoded_output = self.decode_output(model_outputs)
        phonemes = decoded_output["phonemes"]
        phonemes = self._remove_eos(phonemes)
        if single:
            phonemes = phonemes[0]
        return phonemes

    def _remove_eos(self, phonemes):
        """Removes the EOS character from the end of the sequence,
        if encountered

        Arguments
        ---------
        phonemes : list
            a list of phomemic transcriptions

        Returns
        -------
        result : list
            phonemes, without <eos>
        """
        return [
            item[:-1] if item and item[-1] == "<eos>" else item
            for item in phonemes
        ]

    def _update_graphemes(self, model_inputs):
        grapheme_sequence_mode = self.hparams.grapheme_sequence_mode
        if grapheme_sequence_mode and grapheme_sequence_mode != "raw":
            grapheme_encoded_key = f"grapheme_encoded_{grapheme_sequence_mode}"
            if grapheme_encoded_key in model_inputs:
                model_inputs["grapheme_encoded"] = model_inputs[
                    grapheme_encoded_key
                ]

    def load_dependencies(self):
        """Loads any relevant model dependencies"""
        deps_pretrainer = getattr(self.hparams, "deps_pretrainer", None)
        if deps_pretrainer:
            deps_pretrainer.collect_files()
            deps_pretrainer.load_collected()

    def __call__(self, text):
        """A convenience callable wrapper - same as G2P

        Arguments
        ---------
        text: str or list[str]
            a single string to be encoded to phonemes - or a
            sequence of strings

        Returns
        -------
        result: list
            if a single example was provided, the return value is a
            single list of phonemes
        """
        return self.g2p(text)

    def forward(self, noisy, lengths=None):
        """Runs enhancement on the noisy input"""
        return self.enhance_batch(noisy, lengths)


class ResponseGenerator(Pretrained):
    """A ready-to-use Response Generator  model

    The class can be used to generate and continue dialogue given the user input.
    The given YAML must contain the fields specified in the *_NEEDED[] lists.
    It needs to be used with custom.py to load the expanded  model with added tokens like bos,eos, and speaker's tokens.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.
    """

    MODULES_NEEDED = ["model"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #  Load model
        self.model = self.hparams.model
        self.tokenizer = self.model.tokenizer
        self.history_window = 2 * self.hparams.max_history + 1
        self.history = []

    def generate_response(self, turn):
        """
        Complete a dialogue given the user's input.
        Arguments
        ---------
        turn: str
            User input which is the last turn of the dialogue.

        Returns
        -------
        response
            Generated response for the user input based on the dialogue history.
        """

        self.history.append(turn)
        inputs = self.prepare_input()
        hyps = self.generate(inputs)
        predicted_words = self.model.tokenizer.batch_decode(
            hyps[:, inputs[0].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        response = predicted_words[0]
        self.history.append(response)
        return response

    def prepare_input(self):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError

    def generate(self):
        """Users should modify this function according to their own tasks."""
        raise NotImplementedError


class GPTResponseGenerator(ResponseGenerator):
    """A ready-to-use Response Generator  model

    The class can be used to generate and continue dialogue given the user input.
    The given YAML must contain the fields specified in the *_NEEDED[] lists.
    It needs to be used with custom.py to load the expanded GPT model with added tokens like bos,eos, and speaker's tokens.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.text import GPTResponseGenerator

    >>> tmpdir = getfixture("tmpdir")
    >>> res_gen_model = GPTResponseGenerator.from_hparams(
    ...     source="speechbrain/MultiWOZ-GPT-Response_Generation",
    ...     pymodule_file="custom.py",
    ... )  # doctest: +SKIP
    >>> response = res_gen_model.generate_response(
    ...     "I want to book a table for dinner"
    ... )  # doctest: +SKIP
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # convert special tokens to their ids
        (
            self.bos,
            self.eos,
            self.system,
            self.user,
        ) = self.model.tokenizer.convert_tokens_to_ids(
            self.hparams.special_tokens
        )

    def generate(self, inputs):
        """
        Complete a dialogue given the user's input.

        Arguments
        ---------
        inputs: tuple
            history_bos which is the tokenized history+input values with appropriate speaker token appended before each turn and history_token_type which determines
            the type of each token based on who is uttered that token (either User or System).

        Returns
        -------
        response
            Generated hypothesis for the user input based on the dialogue history.
        """

        history_bos, history_token_type = inputs
        padding_mask = ~self.hparams.padding_mask(
            history_bos, pad_idx=self.model.tokenizer.unk_token_id
        )
        hyps = self.model.generate(
            history_bos.detach(),
            history_token_type.detach(),
            padding_mask.detach(),
            "beam",
        )
        return hyps

    def prepare_input(self):
        """Convert user input and previous histories to the format acceptable for  GPT model.
            It appends all previous history and input and truncates it based on max_history value.
            It then tokenizes the input and generates additional input that determines the type of each token (System or User).

        Returns
        -------
        history_bos: torch.Tensor
            Tokenized history+input values with appropriate speaker token appended before each turn.
        history_token_type: torch.LongTensor
            Type of each token based on who is uttered that token (either User or System)
        """
        history_tokens_lists = [
            self.model.tokenizer.encode(turn) for turn in self.history
        ]
        # add speaker tokens to the history turns (user is even, system is odd)
        # BEFORE:  [Hi how are you?], [I'm fine, thanks]
        # AFTER:   [SPK_1 Hi how are you?], [SPK_2 I'm fine, thanks]
        history_input_lists = [
            [self.user if i % 2 == 0 else self.system] + encoded_turn
            for i, encoded_turn in enumerate(history_tokens_lists)
        ]
        history_ids = history_input_lists[-self.history_window :]
        # concatenate every token into a single list
        # list(chain(*[[1, 2], [3, 4], [5]]))
        # >>> [1, 2, 3, 4, 5]
        history_ids = torch.LongTensor(list(chain(*history_ids)))
        # create bos version for the input
        history_bos = torch.cat(
            (torch.tensor([self.bos]), history_ids, torch.tensor([self.system]))
        )
        # create a mapping that associates each token in the input to a speaker
        # INPUT: [SPK_1 Hi    how   are   you? ], [SPK_2 I'm   fine, thanks]
        # TYPE:  [SPK_1 SPK_1 SPK_1 SPK_1 SPK_1], [SPK_2 SPK_2 SPK_2 SPK_2 ]
        history_token_type_lists = [
            [self.user if i % 2 == 0 else self.system] * len(encoded_turn)
            for i, encoded_turn in enumerate(history_input_lists)
        ]
        history_token_type = torch.LongTensor(
            list(
                chain(
                    *(
                        [[self.system]]
                        + history_token_type_lists[-self.history_window :]
                        + [[self.system]]
                    )
                )
            )
        )
        return history_bos.unsqueeze(0), history_token_type.unsqueeze(0)


class Llama2ResponseGenerator(ResponseGenerator):
    """A ready-to-use Response Generator  model

    The class can be used to generate and continue dialogue given the user input.
    The given YAML must contain the fields specified in the *_NEEDED[] lists.
    It needs to be used with custom.py to load the expanded Llama2 model with added tokens like bos,eos, and speaker's tokens.

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> from speechbrain.inference.text import Llama2ResponseGenerator

    >>> tmpdir = getfixture("tmpdir")
    >>> res_gen_model = Llama2ResponseGenerator.from_hparams(
    ...     source="speechbrain/MultiWOZ-Llama2-Response_Generation",
    ...     pymodule_file="custom.py",
    ... )  # doctest: +SKIP
    >>> response = res_gen_model.generate_response(
    ...     "I want to book a table for dinner"
    ... )  # doctest: +SKIP
    """

    def __init__(self, *args, **kwargs):
        run_opts = {"device": "cuda"}
        super().__init__(run_opts=run_opts, *args, **kwargs)
        # self.model = self.model#.to("cuda")

    def generate(self, inputs):
        """
        Complete a dialogue given the user's input.
        Arguments
        ---------
        inputs: prompt_bos
            prompted inputs to be passed to llama2 model for generation.

        Returns
        -------
        response
            Generated hypothesis for the user input based on the dialogue history.
        """
        prompt_bos = inputs[0].to(self.model.model.device)
        padding_mask = ~self.hparams.padding_mask(
            prompt_bos, pad_idx=self.tokenizer.pad_token_id
        )
        hyps = self.model.generate(
            prompt_bos.detach(),
            padding_mask.detach(),
            "beam",
        )
        return hyps

    def prepare_input(self):
        """Convert user input and previous histories to the format acceptable for  Llama2 model.
            It appends all previous history and input and truncates it based on max_history value.
            It then tokenizes the input and add prompts.

        Returns
        -------
        prompt_bos: torch.Tensor
            Tokenized history+input values with appropriate prompt.
        """

        def generate_prompt(idx_and_item):
            """add [INST] and [/INST] prompt to the start and end ogf item.

            Arguments
            ---------
            idx_and_item: tuple
                id and its corresponding text. If the id is even, it is user turn and [ INST] is added.

            Returns
            -------
            prompt_bos: torch.LongTensor
                prompted text for one item.
            """
            index, item = idx_and_item
            if index % 2 == 0:
                return "[INST] " + item + " [/INST]"
            else:
                return item

        prompts = list(map(generate_prompt, enumerate(self.history)))

        # encode each turn of the history
        prompt_tokens_lists = [self.tokenizer.encode(turn) for turn in prompts]

        prompt_ids = prompt_tokens_lists[-self.history_window :]
        # concatenate every token into a single list
        # list(chain(*[[1, 2], [3, 4], [5]]))
        # >>> [1, 2, 3, 4, 5]
        prompt_ids = torch.LongTensor(list(chain(*prompt_ids)))
        # without bos for lm_labels

        # # create bos version for the input
        prompt_bos = torch.cat(
            (torch.tensor([self.tokenizer.bos_token_id]), prompt_ids)
        )
        return prompt_bos.unsqueeze(0).unsqueeze(dim=0)
