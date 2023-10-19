"""
Neural network modules for the Zero-Shot Multi-Speaker Tacotron2 end-to-end neural
Text-to-Speech (TTS) model

Authors
* Georges Abous-Rjeili 2021
* Artem Ploujnikov 2021
* Pradnya Kandarkar 2023
"""

# This code uses a significant portion of the NVidia implementation, even though it
# has been modified and enhanced

# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/model.py
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from math import sqrt
from speechbrain.nnet.loss.guidedattn_loss import GuidedAttentionLoss
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import pickle
from speechbrain.lobes.models.Tacotron2 import (
    LinearNorm,
    Postnet,
    Encoder,
    Decoder,
    get_mask_from_lengths,
)


class Tacotron2(nn.Module):
    """The Tactron2 text-to-speech model, based on the NVIDIA implementation.

    This class is the main entry point for the model, which is responsible
    for instantiating all submodules, which, in turn, manage the individual
    neural network layers

    Simplified STRUCTURE: phoneme input->token embedding ->encoder -> (encoder output + speaker embedding) ->attention \
    ->decoder(+prenet) -> postnet ->output

    prenet(input is decoder previous time step) output is input to decoder
    concatenanted with the attention output

    Arguments
    ---------
    spk_emb_size: int
        Speaker embedding size

    mask_padding: bool
        whether or not to mask pad-outputs of tacotron

    #mel generation parameter in data io
    n_mel_channels: int
        number of mel channels for constructing spectrogram

    #symbols
    n_symbols:  int=128
        number of accepted char symbols defined in textToSequence
    symbols_embedding_dim: int
        number of embeding dimension for symbols fed to nn.Embedding

    # Encoder parameters
    encoder_kernel_size: int
        size of kernel processing the embeddings
    encoder_n_convolutions: int
        number of convolution layers in encoder
    encoder_embedding_dim: int
        number of kernels in encoder, this is also the dimension
        of the bidirectional LSTM in the encoder

    # Attention parameters
    attention_rnn_dim: int
        input dimension
    attention_dim: int
        number of hidden represetation in attention
    # Location Layer parameters
    attention_location_n_filters: int
        number of 1-D convulation filters in attention
    attention_location_kernel_size: int
        length of the 1-D convolution filters

    # Decoder parameters
    n_frames_per_step: int=1
        only 1 generated mel-frame per step is supported for the decoder as of now.
    decoder_rnn_dim: int
        number of 2 unidirectionnal stacked LSTM units
    prenet_dim: int
        dimension of linear prenet layers
    max_decoder_steps: int
        maximum number of steps/frames the decoder generates before stopping
    p_attention_dropout: float
        attention drop out probability
    p_decoder_dropout: float
        decoder drop  out probability

    gate_threshold: int
        cut off level any output probabilty above that is considered
        complete and stops genration so we have variable length outputs
    decoder_no_early_stopping: bool
        determines early stopping of decoder
        along with gate_threshold . The logical inverse of this is fed to the decoder


    #Mel-post processing network parameters
    postnet_embedding_dim: int
        number os postnet dfilters
    postnet_kernel_size: int
        1d size of posnet kernel
    postnet_n_convolutions: int
        number of convolution layers in postnet

    Example
    -------
    >>> import torch
    >>> _ = torch.manual_seed(213312)
    >>> from speechbrain.lobes.models.Tacotron2 import Tacotron2
    >>> model = Tacotron2(
    ...    mask_padding=True,
    ...    n_mel_channels=80,
    ...    n_symbols=148,
    ...    symbols_embedding_dim=512,
    ...    encoder_kernel_size=5,
    ...    encoder_n_convolutions=3,
    ...    encoder_embedding_dim=512,
    ...    attention_rnn_dim=1024,
    ...    attention_dim=128,
    ...    attention_location_n_filters=32,
    ...    attention_location_kernel_size=31,
    ...    n_frames_per_step=1,
    ...    decoder_rnn_dim=1024,
    ...    prenet_dim=256,
    ...    max_decoder_steps=32,
    ...    gate_threshold=0.5,
    ...    p_attention_dropout=0.1,
    ...    p_decoder_dropout=0.1,
    ...    postnet_embedding_dim=512,
    ...    postnet_kernel_size=5,
    ...    postnet_n_convolutions=5,
    ...    decoder_no_early_stopping=False
    ... )
    >>> _ = model.eval()
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> outputs, output_lengths, alignments = model.infer(inputs, input_lengths)
    >>> outputs.shape, output_lengths.shape, alignments.shape
    (torch.Size([2, 80, 1]), torch.Size([2]), torch.Size([2, 1, 5]))
    """

    def __init__(
        self,
        spk_emb_size,
        mask_padding=True,
        n_mel_channels=80,
        n_symbols=148,
        symbols_embedding_dim=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        attention_rnn_dim=1024,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        decoder_no_early_stopping=False,
    ):
        super().__init__()
        self.mask_padding = mask_padding
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(
            encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size
        )
        self.decoder = Decoder(
            n_mel_channels,
            n_frames_per_step,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_rnn_dim,
            decoder_rnn_dim,
            prenet_dim,
            max_decoder_steps,
            gate_threshold,
            p_attention_dropout,
            p_decoder_dropout,
            not decoder_no_early_stopping,
        )
        self.postnet = Postnet(
            n_mel_channels,
            postnet_embedding_dim,
            postnet_kernel_size,
            postnet_n_convolutions,
        )

        # Additions for Zero-Shot Multi-Speaker TTS
        # FiLM (Feature-wise Linear Modulation) layers for injecting the speaker embeddings into the TTS pipeline
        self.ms_film_hidden_size = int(
            (spk_emb_size + encoder_embedding_dim) / 2
        )
        self.ms_film_hidden = LinearNorm(spk_emb_size, self.ms_film_hidden_size)
        self.ms_film_h = LinearNorm(
            self.ms_film_hidden_size, encoder_embedding_dim
        )
        self.ms_film_g = LinearNorm(
            self.ms_film_hidden_size, encoder_embedding_dim
        )

    def parse_output(self, outputs, output_lengths, alignments_dim=None):
        """
        Masks the padded part of output

        Arguments
        ---------
        outputs: list
            a list of tensors - raw outputs
        output_lengths: torch.Tensor
            a tensor representing the lengths of all outputs
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training


        Returns
        -------
        result: tuple
            a (mel_outputs, mel_outputs_postnet, gate_outputs, alignments) tuple with
            the original outputs - with the mask applied
        """
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = outputs
        if self.mask_padding and output_lengths is not None:
            mask = get_mask_from_lengths(
                output_lengths, max_len=mel_outputs.size(-1)
            )
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_outputs.clone().masked_fill_(mask, 0.0)
            mel_outputs_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)  # gate energies
        if alignments_dim is not None:
            alignments = F.pad(
                alignments, (0, alignments_dim - alignments.size(-1))
            )

        return (
            mel_outputs,
            mel_outputs_postnet,
            gate_outputs,
            alignments,
            output_lengths,
        )

    def forward(self, inputs, spk_embs, alignments_dim=None):
        """Decoder forward pass for training

        Arguments
        ---------
        inputs: tuple
            batch object
        spk_embs: torch.Tensor
            Speaker embeddings corresponding to the inputs
        alignments_dim: int
            the desired dimension of the alignments along the last axis
            Optional but needed for data-parallel training

        Returns
        ---------
        mel_outputs: torch.Tensor
            mel outputs from the decoder
        mel_outputs_postnet: torch.Tensor
            mel outputs from postnet
        gate_outputs: torch.Tensor
            gate outputs from the decoder
        alignments: torch.Tensor
            sequence of attention weights from the decoder
        output_legnths: torch.Tensor
            length of the output without padding
        """
        inputs, input_lengths, targets, max_len, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        # Inject speaker embeddings into the encoder output
        spk_embs_shared = F.relu(self.ms_film_hidden(spk_embs))

        spk_embs_h = self.ms_film_h(spk_embs_shared)
        spk_embs_h = torch.unsqueeze(spk_embs_h, 1).repeat(
            1, encoder_outputs.shape[1], 1
        )
        encoder_outputs = encoder_outputs * spk_embs_h

        spk_embs_g = self.ms_film_g(spk_embs_shared)
        spk_embs_g = torch.unsqueeze(spk_embs_g, 1).repeat(
            1, encoder_outputs.shape[1], 1
        )
        encoder_outputs = encoder_outputs + spk_embs_g

        # Pass the encoder output combined with speaker embeddings to the next layers
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths,
            alignments_dim,
        )

    def infer(self, inputs, spk_embs, input_lengths):
        """Produces outputs


        Arguments
        ---------
        inputs: torch.tensor
            text or phonemes converted
        spk_embs: torch.Tensor
            Speaker embeddings corresponding to the inputs
        input_lengths: torch.tensor
            the lengths of input parameters

        Returns
        -------
        mel_outputs_postnet: torch.Tensor
            final mel output of tacotron 2
        mel_lengths: torch.Tensor
            length of mels
        alignments: torch.Tensor
            sequence of attention weights
        """

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.infer(embedded_inputs, input_lengths)

        # Inject speaker embeddings into the encoder output
        spk_embs_shared = F.relu(self.ms_film_hidden(spk_embs))

        spk_embs_h = self.ms_film_h(spk_embs_shared)
        spk_embs_h = torch.unsqueeze(spk_embs_h, 1).repeat(
            1, encoder_outputs.shape[1], 1
        )
        encoder_outputs = encoder_outputs * spk_embs_h

        spk_embs_g = self.ms_film_g(spk_embs_shared)
        spk_embs_g = torch.unsqueeze(spk_embs_g, 1).repeat(
            1, encoder_outputs.shape[1], 1
        )
        encoder_outputs = encoder_outputs + spk_embs_g

        # Pass the encoder output combined with speaker embeddings to the next layers
        mel_outputs, gate_outputs, alignments, mel_lengths = self.decoder.infer(
            encoder_outputs, input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        BS = mel_outputs_postnet.size(0)
        alignments = alignments.unfold(1, BS, BS).transpose(0, 2)

        return mel_outputs_postnet, mel_lengths, alignments


LossStats = namedtuple(
    "TacotronLoss", "loss mel_loss spk_emb_loss gate_loss attn_loss attn_weight"
)


class Loss(nn.Module):
    """The Tacotron loss implementation
    The loss consists of an MSE loss on the spectrogram, a BCE gate loss
    and a guided attention loss (if enabled) that attempts to make the
    attention matrix diagonal
    The output of the moduel is a LossStats tuple, which includes both the
    total loss
    Arguments
    ---------
    guided_attention_sigma: float
        The guided attention sigma factor, controling the "width" of
        the mask
    gate_loss_weight: float
        The constant by which the gate loss will be multiplied
    mel_loss_weight: float
        The constant by which the mel loss will be multiplied
    spk_emb_loss_weight: float
        The constant by which the speaker embedding loss will be multiplied - placeholder for future work
    spk_emb_loss_type: str
        Type of the speaker embedding loss - placeholder for future work
    guided_attention_weight: float
        The weight for the guided attention
    guided_attention_scheduler: callable
        The scheduler class for the guided attention loss
    guided_attention_hard_stop: int
        The number of epochs after which guided attention will be compeltely
        turned off
    Example:
    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> from speechbrain.lobes.models.MSTacotron2 import Loss
    >>> loss = Loss(guided_attention_sigma=0.2)
    >>> mel_target = torch.randn(2, 80, 861)
    >>> gate_target = torch.randn(1722, 1)
    >>> mel_out = torch.randn(2, 80, 861)
    >>> mel_out_postnet = torch.randn(2, 80, 861)
    >>> gate_out = torch.randn(2, 861)
    >>> alignments = torch.randn(2, 861, 173)
    >>> pred_mel_lens = torch.randn(2)
    >>> targets = mel_target, gate_target
    >>> model_outputs = mel_out, mel_out_postnet, gate_out, alignments, pred_mel_lens
    >>> input_lengths = torch.tensor([173,  91])
    >>> target_lengths = torch.tensor([861, 438])
    >>> spk_embs = None
    >>> loss(model_outputs, targets, input_lengths, target_lengths, spk_embs, 1)
    TacotronLoss(loss=tensor([4.8566]), mel_loss=tensor(4.0097), spk_emb_loss=tensor([0.]), gate_loss=tensor(0.8460), attn_loss=tensor(0.0010), attn_weight=tensor(1.))
    """

    def __init__(
        self,
        guided_attention_sigma=None,
        gate_loss_weight=1.0,
        mel_loss_weight=1.0,
        spk_emb_loss_weight=1.0,
        spk_emb_loss_type=None,
        guided_attention_weight=1.0,
        guided_attention_scheduler=None,
        guided_attention_hard_stop=None,
    ):
        super().__init__()
        if guided_attention_weight == 0:
            guided_attention_weight = None
        self.guided_attention_weight = guided_attention_weight
        self.gate_loss_weight = gate_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.spk_emb_loss_weight = spk_emb_loss_weight
        self.spk_emb_loss_type = spk_emb_loss_type

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.guided_attention_loss = GuidedAttentionLoss(
            sigma=guided_attention_sigma
        )
        self.cos_sim = nn.CosineSimilarity()
        self.triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
        )
        self.cos_emb_loss = nn.CosineEmbeddingLoss()

        self.guided_attention_scheduler = guided_attention_scheduler
        self.guided_attention_hard_stop = guided_attention_hard_stop

    def forward(
        self,
        model_output,
        targets,
        input_lengths,
        target_lengths,
        spk_embs,
        epoch,
    ):
        """Computes the loss
        Arguments
        ---------
        model_output: tuple
            the output of the model's forward():
            (mel_outputs, mel_outputs_postnet, gate_outputs, alignments)
        targets: tuple
            the targets
        input_lengths: torch.Tensor
            a (batch, length) tensor of input lengths
        target_lengths: torch.Tensor
            a (batch, length) tensor of target (spectrogram) lengths
        spk_embs: torch.Tensor
            Speaker embedding input for the loss computation - placeholder for future work
        epoch: int
            the current epoch number (used for the scheduling of the guided attention
            loss) A StepScheduler is typically used
        Returns
        -------
        result: LossStats
            the total loss - and individual losses (mel and gate)
        """
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        (
            mel_out,
            mel_out_postnet,
            gate_out,
            alignments,
            pred_mel_lens,
        ) = model_output

        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + self.mse_loss(
            mel_out_postnet, mel_target
        )

        mel_loss = self.mel_loss_weight * mel_loss

        gate_loss = self.gate_loss_weight * self.bce_loss(gate_out, gate_target)
        attn_loss, attn_weight = self.get_attention_loss(
            alignments, input_lengths, target_lengths, epoch
        )

        # Speaker embedding loss placeholder - for future work
        spk_emb_loss = torch.Tensor([0]).to(mel_loss.device)

        if self.spk_emb_loss_type == "scl_loss":
            target_spk_embs, preds_spk_embs = spk_embs

            cos_sim_scores = self.cos_sim(preds_spk_embs, target_spk_embs)
            spk_emb_loss = -torch.div(
                torch.sum(cos_sim_scores), len(cos_sim_scores)
            )

        if self.spk_emb_loss_type == "cos_emb_loss":
            target_spk_embs, preds_spk_embs = spk_embs
            spk_emb_loss = self.cos_emb_loss(
                target_spk_embs,
                preds_spk_embs,
                torch.ones(len(target_spk_embs)).to(target_spk_embs.device),
            )

        if self.spk_emb_loss_type == "triplet_loss":
            anchor_spk_embs, pos_spk_embs, neg_spk_embs = spk_embs
            if anchor_spk_embs is not None:
                spk_emb_loss = self.triplet_loss(
                    anchor_spk_embs, pos_spk_embs, neg_spk_embs
                )

        spk_emb_loss = self.spk_emb_loss_weight * spk_emb_loss

        total_loss = mel_loss + spk_emb_loss + gate_loss + attn_loss
        return LossStats(
            total_loss,
            mel_loss,
            spk_emb_loss,
            gate_loss,
            attn_loss,
            attn_weight,
        )

    def get_attention_loss(
        self, alignments, input_lengths, target_lengths, epoch
    ):
        """Computes the attention loss
        Arguments
        ---------
        alignments: torch.Tensor
            the aligment matrix from the model
        input_lengths: torch.Tensor
            a (batch, length) tensor of input lengths
        target_lengths: torch.Tensor
            a (batch, length) tensor of target (spectrogram) lengths
        epoch: int
            the current epoch number (used for the scheduling of the guided attention
            loss) A StepScheduler is typically used
        Returns
        -------
        attn_loss: torch.Tensor
            the attention loss value
        """
        zero_tensor = torch.tensor(0.0, device=alignments.device)
        if (
            self.guided_attention_weight is None
            or self.guided_attention_weight == 0
        ):
            attn_weight, attn_loss = zero_tensor, zero_tensor
        else:
            hard_stop_reached = (
                self.guided_attention_hard_stop is not None
                and epoch > self.guided_attention_hard_stop
            )
            if hard_stop_reached:
                attn_weight, attn_loss = zero_tensor, zero_tensor
            else:
                attn_weight = self.guided_attention_weight
                if self.guided_attention_scheduler is not None:
                    _, attn_weight = self.guided_attention_scheduler(epoch)
            attn_weight = torch.tensor(attn_weight, device=alignments.device)
            attn_loss = attn_weight * self.guided_attention_loss(
                alignments, input_lengths, target_lengths
            )
        return attn_loss, attn_weight


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    Arguments
    ---------
    speaker_embeddings_pickle : str
        Path to the file containing speaker embeddings
    n_frames_per_step: int
        The number of output frames per step
    Returns
    -------
    result: tuple
        a tuple inputs/targets
        (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            spk_embs,
            spk_ids
        )
    """

    def __init__(
        self, speaker_embeddings_pickle, n_frames_per_step=1,
    ):
        self.n_frames_per_step = n_frames_per_step
        self.speaker_embeddings_pickle = speaker_embeddings_pickle

    # TODO: Make this more intuitive, use the pipeline
    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        Arguments
        ---------
        batch: list
            [text_normalized, mel_normalized]
        """

        # TODO: Remove for loops and this dirty hack
        raw_batch = list(batch)
        for i in range(
            len(batch)
        ):  # the pipline return a dictionary wiht one elemnent
            batch[i] = batch[i]["mel_text_pair"]

        # Right zero-pad all one-hot text sequences to max input length

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        labels, wavs, spk_embs_list, spk_ids = [], [], [], []
        with open(
            self.speaker_embeddings_pickle, "rb"
        ) as speaker_embeddings_file:
            speaker_embeddings = pickle.load(speaker_embeddings_file)

        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)
            labels.append(raw_batch[idx]["label"])
            wavs.append(raw_batch[idx]["wav"])

            spk_emb = speaker_embeddings[raw_batch[idx]["uttid"]]
            spk_embs_list.append(spk_emb)

            spk_ids.append(raw_batch[idx]["uttid"].split("_")[0])

        spk_embs = torch.stack(spk_embs_list)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
            len_x,
            labels,
            wavs,
            spk_embs,
            spk_ids,
        )
