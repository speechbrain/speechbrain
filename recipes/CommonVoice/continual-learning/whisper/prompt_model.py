"""This lobe enables the integration of huggingface pretrained whisper model with prompt-based cl.


Authors
 * Pooneh Mousavi 2023

"""

import random
import math
import torch
import torch.utils.checkpoint
from torch import nn
import logging
import copy


from transformers.models.whisper.tokenization_whisper import (
    LANGUAGES,
    TASK_IDS,
    TO_LANGUAGE_CODE,
    WhisperTokenizer,
)

from transformers.models.whisper.modeling_whisper import (
        WhisperPositionalEmbedding,
        WhisperDecoder,
        WhisperDecoderLayer,
    )

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper

logger = logging.getLogger(__name__)

class ProgressiveWhisperTokenizer(WhisperTokenizer):
    # override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_languages = {k: v for k, v in LANGUAGES.items()}  # Copy
        self.to_language_codes = {
            k: v for k, v in TO_LANGUAGE_CODE.items()
        }  # Copy

    # override
    @property
    def prefix_tokens(self):
        # all_special_ids = self.all_special_ids
        bos_token_id = 50258  # all_special_ids[-106]
        translate_token_id = 50358  # all_special_ids[-6]
        transcribe_token_id = 50359  # all_special_ids[-5]
        notimestamps_token_id = 50363  # all_special_ids[-1]
        # langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in self.to_language_codes:
                language_id = self.to_language_codes[self.language]
            else:
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be in: {self.to_language_codes.keys()}"
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(
                    f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}"
                )

        bos_sequence = [bos_token_id]
        if self.language is not None:
            # Need to replace with custom code because language ID is hardcoded...
            # bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
            bos_sequence.append(
                self.encode(f"<|{language_id}|>", add_special_tokens=False)[0]
            )
        if self.task is not None:
            bos_sequence.append(
                transcribe_token_id
                if self.task == "transcribe"
                else translate_token_id
            )
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence


class PromptWhisperDecoder(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def __init__(
        self, source,
    ):
        super().__init__(
            source
        )

    # override
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prompt=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            Prompt: Dic
               dic contains information about computed prompts  for the input.It contains prompts, length of added prompts and similarity scores for learning keys.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if prompt:
            encoder_hidden_states,inputs_embeds, input_ids= self.compute_prompted_input(audio_features=encoder_hidden_states, inputs_embeds=inputs_embeds,input_ids=input_ids,prompt_out=prompt)
            input_shape = input_ids.size()



        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache ="
                        " False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
    def compute_prompted_input(self,audio_features, inputs_embeds,input_ids,prompt_out):
        batched_prompt= prompt_out['batched_prompt']
        if prompt_out['prompt_loc_mode'] == 'enc' or prompt_out['prompt_loc_mode'] == 'both':
            audio_features = torch.cat([batched_prompt, audio_features], dim=1)
        if prompt_out['prompt_loc_mode'] == 'dec' or prompt_out['prompt_loc_mode'] == 'both':
            inputs_embeds = torch.cat([inputs_embeds[:,:4,:],batched_prompt, inputs_embeds[:,4:,:]], dim=1)
            input_ids = torch.cat([input_ids[:,:4],torch.full((input_ids.shape[0],batched_prompt.shape[1]),prompt_out['token_id']).to(input_ids.get_device()), input_ids[:,4:]], dim=1)
        return audio_features,inputs_embeds,input_ids








class PromptWhisper(HuggingFaceWhisper):
    # override
    def __init__(
        self, source,prompt,freeze_blocks, save_path,prompt_loc_mode='enc',prompt_enabled=True,**kwargs,
    ):
        """Perform mel transformation and one step of the whisper (encoder-decoder) with prompt learning.

            Arguments
            ---------
            config: WhisperConfig
            prompt : Prompt
                A Prompt Module contains prompt pool.
            prompt_loc_mode: str
                where to add prompts (enc: add prompts as audio-features, dec: add prompts as decoder_input_ids and  both. Prompts are added at time-dimension)
            prompt_enabled : boolean
                wheter to use prompt or not.
            freeze_blocks list[str]:
                indictaed the blocks that need to be frozen in the whisper model.


            """

        super().__init__(
            source, save_path, **kwargs,
        )
        if self.tokenizer is not None:
            self.tokenizer = ProgressiveWhisperTokenizer.from_pretrained(
                source,
                language=None,
                task="transcribe",
                predict_timestamps=False,
            )
            # The number of embeddings is 51865 while the vocabulary size is 50364
            # The missing tokens are timestamp tokens (see https://github.com/openai/whisper/discussions/361)
            # To avoid problems when extending the tokenizer and/or the model we add them explicitly
            vocab_size = len(self.tokenizer.get_vocab())
            
            num_embeddings = self.model.decoder.embed_tokens.num_embeddings
            num_missing_tokens = num_embeddings - vocab_size
            timestamps = [
                i * 30.0 / (num_missing_tokens - 1)
                for i in range(num_missing_tokens)
            ]
            timestamp_tokens = [f"<|{ts:.2f}|>" for ts in timestamps]
            self.tokenizer.add_tokens(timestamp_tokens)
        
        
        self.prompt=prompt
        self.prompt_loc_mode=prompt_loc_mode
        self.prompt_enabled=prompt_enabled
        self.freeze_blocks=freeze_blocks
        self.model.decoder.save_pretrained("./decoder")
        self.model.decoder=PromptWhisperDecoder.from_pretrained("./decoder")
        self.set_require_grad(self.freeze_blocks)



    def set_require_grad(self,freeze_blocks):
        for n, p in self.named_parameters():
            if n.startswith(tuple(freeze_blocks)):
                p.requires_grad = False
            else:
                p.requires_grad = True
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('number of params:', n_parameters)


    def forward(self, wav, decoder_input_ids=None):
        """Perform mel transformation and one step of the whisper (encoder-decoder).

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        decoder_input_ids : torch.Tensor
            This is necessary if we want to use the decoder.

            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.

            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.

        """
        out_encoder = self.forward_encoder(wav)
        if self.prompt_enabled:
            prompt_out= self.prompt(out_encoder, prompt_mask=None)
            prompt_out['prompt_loc_mode']=self.prompt_loc_mode
        else:
            prompt_out=None
                

        if self.output_all_hiddens:
            logits, attn = self.forward_decoder(
                out_encoder[-1], decoder_input_ids,prompt=prompt_out
            )
        else:
            logits, attn = self.forward_decoder(
                out_encoder, decoder_input_ids,prompt=prompt_out
            )
        return out_encoder, logits, attn, prompt_out




    
    
    
    def forward_decoder(self, audio_features, decoder_input_ids,prompt=None):
        """Perform one step of the whisper decoder.
        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features (mel + whisper encoding).
        decoder_input_ids : torch.Tensor
            A batch of decoder inputs tokens.
            The first tokens need to dictacte the behavior of the decoder.
            It needs to start with the bos_token, the language token,
            the task token, and finally the timestamp token.

            Please refer to the whisper paper for more details or go to the
            seq2seq2.py file in SpeechBrain to see how to generate the tokens
            with Greedy Search and/or Beam Search.
        Prompt: Dic
          dic contains information about computed prompts  for the input.It contains prompts, length of added prompts and similarity scores for learning keys.

        
        """
        output_states = self.model.decoder(
            encoder_hidden_states=audio_features,
            input_ids=decoder_input_ids,
            output_attentions=self.output_attentions,
            prompt=prompt
        )

        attn = output_states.attentions[-1]
        attn = attn.view(attn.shape[0] * attn.shape[1], *attn.shape[2:])
        output_states = output_states.last_hidden_state

        logits = (
            output_states
            @ torch.transpose(
                self.model.decoder.embed_tokens.weight.to(output_states.dtype),
                0,
                1,
            )
        ).to(audio_features.dtype)

        return logits, attn

    

    @torch.no_grad()
    def generate(
        self,
        wav=None,
        audio_features=None,
        forced_decoder_locale=None,
        max_gen_tokens=445,
        strategy="greedy",
    ):
        if wav is None and audio_features is None:
            raise ValueError(
                "Either `wav` or `audio_features` argument should be given"
            )
        if audio_features is None:
            audio_features = self.forward_encoder(wav)
        
        if self.prompt_enabled:
            prompt_out= self.prompt(audio_features, prompt_mask=None)
            prompt_out['prompt_loc_mode']=self.prompt_loc_mode
        else:
            prompt_out=None


        batch_size = audio_features.shape[0]
        (
            startoftranscript_id,
            transcribe_id,
            notimestamps_id,
        ) = self.tokenizer.prefix_tokens
        pad_id = self.model.config.pad_token_id
        endoftext_id = self.tokenizer.eos_token_id

        hyps = torch.full(
            (batch_size, max_gen_tokens  + 4),
            pad_id,
            dtype=torch.long,
            device=audio_features.device,
        )
        if forced_decoder_locale is None:
            # Compute most likely language token IDs
            all_lang_tokens = [
                f"<|{l}|>" for l in self.tokenizer.supported_languages
            ]
            all_lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                all_lang_tokens
            )
            hyps[:, 0] = startoftranscript_id
            logits, _ = self.forward_decoder(audio_features, hyps[:, :1])
            lang_mask = torch.zeros(
                logits.shape[-1], device=logits.device, dtype=torch.bool
            )
            lang_mask[all_lang_tokens_ids] = True
            logits[:, :, ~lang_mask] = -float("inf")
            lang_tokens_ids = logits.argmax(dim=-1)[:, 0]
        else:
            if forced_decoder_locale.lower() == "zh-cn":
                forced_decoder_locale = "zh"
            if (
                forced_decoder_locale.lower()
                not in self.tokenizer.supported_languages
            ):
                raise NotImplementedError(
                    f"Unsupported language: {forced_decoder_locale}"
                )
            lang_tokens_ids = self.tokenizer.convert_tokens_to_ids(
                f"<|{forced_decoder_locale.lower()}|>"
            )

        # Prepare initial tokens in the right format
        hyps[:, 0] = startoftranscript_id
        hyps[:, 1] = lang_tokens_ids
        hyps[:, 2] = transcribe_id
        hyps[:, 3] = notimestamps_id

      

        # Autoregressive loop
        num_gen_tokens = 0
        unfinished_mask = torch.ones(
            len(hyps), dtype=torch.bool, device=audio_features.device
        )
        if self.prompt_enabled:
            prompts=copy.deepcopy(prompt_out['batched_prompt'])
        while True:
            if self.prompt_enabled:
                prompt_out['batched_prompt']=prompts[unfinished_mask]
            logits, _ = self.forward_decoder(
                audio_features[unfinished_mask],
                hyps[unfinished_mask, : num_gen_tokens + 4],prompt=prompt_out
            )
            
            if self.prompt_enabled:
                if (prompt_out['prompt_loc_mode'] == 'dec' or prompt_out['prompt_loc_mode'] == 'both'):
                    splits=torch.split(logits,[4,prompt_out['total_prompt_len'],logits.shape[1]-prompt_out['total_prompt_len']-4],dim=1)
                    logits=torch.cat((splits[0],splits[2]),1)
            # Prepare suppress mask
            suppress_mask = torch.ones(
                logits.shape[-1], device=audio_features.device, dtype=torch.bool
            )
            suppress_mask[self.model.config.suppress_tokens] = False
            logits[:, :, ~suppress_mask] = -float("inf")
            gen_tokens = logits.argmax(dim=-1)[:, -1]
            hyps[unfinished_mask, num_gen_tokens + 4] = gen_tokens
            unfinished_mask[unfinished_mask == True] = (
                gen_tokens != endoftext_id
            )
            num_gen_tokens += 1
            if (not unfinished_mask.any()) or (
                num_gen_tokens >= max_gen_tokens
            ):
                break

        if self.prompt_enabled:
            prompt_out['batched_prompt']=prompts
        return hyps[:, 4 : num_gen_tokens + 3],prompt_out

