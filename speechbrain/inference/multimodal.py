""" Specifies the inference interfaces for multi-modal models such as audio/speech LLM.

Authors:
 * Yingzhi Wang 2024
"""

import torch
import torchaudio

import speechbrain as sb
from speechbrain.inference.interfaces import Pretrained


class LTU_AS(Pretrained):
    """A ready-to-use Audio/Speech LLM inference interface"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tltr = self.hparams.tltr
        self.whisper = self.hparams.whisper
        self.avg_pool = self.hparams.avg_pool
        self.audio_proj = self.hparams.audio_proj
        self.llama3 = self.hparams.llama3
        self.tokenizer = self.llama3.tokenizer
        self.embedding_layer = self.hparams.llama3.model.get_input_embeddings()

        self.whisper = self.whisper.to(self.device)

        # whisper pad/trim all the audios to 10 seconds
        chunked_embed_positions_weight = torch.nn.Parameter(
            self.whisper.model.encoder.embed_positions.weight[:500, :]
        )
        self.whisper.model.encoder.embed_positions.weight = (
            chunked_embed_positions_weight
        )

    def generate_with_raw_audio(self, audio_path, instruction, transcript):
        """
        Follows the user's text instruction based on the input audio.

        Arguments
        ---------
        audio_path: str
            Input audio path.
        instruction: str
            User's instruction to be passed to llama3 model for generation.
        transcript: str
            Audio's transcript (from an ASR model).

        Returns
        -------
        response
            Generated hypothesis for the user input based on the audio.
        """
        info = sb.dataio.dataio.read_audio_info(audio_path)
        sig = sb.dataio.dataio.read_audio(audio_path)

        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=1)

        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            16000,
        )(sig)
        resampled = resampled.unsqueeze(0).to(self.device)

        # get audio embedding
        audio_embs = self.whisper(resampled, n_samples=160000)[1:]
        audio_embs = audio_embs.squeeze()
        audio_embs = self.avg_pool(audio_embs)
        audio_embs = audio_embs.unsqueeze(0)
        audio_embs = self.tltr(audio_embs)
        audio_embs = self.audio_proj(audio_embs)

        # get text embedding
        user_prompt_bos = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an assistant that understands audio and speech.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction} The transcript of the audio is:{transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        input_tokens = self.tokenizer.encode(
            user_prompt_bos,
            truncation=True,
            max_length=200,
            padding=False,
            return_tensors=None,
        )
        input_tokens = torch.LongTensor(input_tokens)
        input_tokens = input_tokens.unsqueeze(0)
        input_tokens = input_tokens.to(self.device)
        input_embed = self.embedding_layer(input_tokens)

        # concat audio and text embedding
        input_embed = torch.concat([audio_embs, input_embed], dim=1)

        # get padding masks for audio and text, then concat
        text_padding_mask = ~self.hparams.text_padding_mask(
            input_tokens, pad_idx=0
        )
        text_padding_mask = text_padding_mask.long()
        audio_padding_mask = torch.ones(
            [text_padding_mask.shape[0], 25], device = self.device
        )
        input_mask = torch.concat(
            [audio_padding_mask, text_padding_mask], dim=1
        )

        # run llama
        hyps = self.llama3.generate(
            inputs_embeds=input_embed.detach(),
            attention_mask=input_mask.detach(),
        )

        predicted_words = self.tokenizer.batch_decode(
            hyps,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return predicted_words
