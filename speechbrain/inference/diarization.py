"""Specifies the inference interfaces for diarization modules.

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

import torch

from speechbrain.inference.interfaces import Pretrained


class Speech_Emotion_Diarization(Pretrained):
    """A ready-to-use SED interface (audio -> emotions and their durations)

    Arguments
    ---------
    See ``Pretrained``

    Example
    -------
    >>> from speechbrain.inference.diarization import Speech_Emotion_Diarization
    >>> tmpdir = getfixture("tmpdir")
    >>> sed_model = Speech_Emotion_Diarization.from_hparams(
    ...     source="speechbrain/emotion-diarization-wavlm-large",
    ...     savedir=tmpdir,
    ... )  # doctest: +SKIP
    >>> sed_model.diarize_file(
    ...     "speechbrain/emotion-diarization-wavlm-large/example.wav"
    ... )  # doctest: +SKIP
    """

    MODULES_NEEDED = ["input_norm", "wav2vec", "output_mlp"]

    def diarize_file(self, path):
        """Get emotion diarization of a spoken utterance.

        Arguments
        ---------
        path : str
            Path to audio file which to diarize.

        Returns
        -------
        list of dictionary: List[Dict[List]]
            The emotions and their temporal boundaries.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        frame_class = self.diarize_batch(batch, rel_length, [path])
        return frame_class

    def encode_batch(self, wavs, wav_lens):
        """Encodes audios into fine-grained emotional embeddings

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels].
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.Tensor
            The encoded batch
        """
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        wavs = self.mods.input_norm(wavs, wav_lens)
        outputs = self.mods.wav2vec2(wavs)
        return outputs

    def diarize_batch(self, wavs, wav_lens, batch_id):
        """Get emotion diarization of a batch of waveforms.

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.Tensor
            Batch of waveforms [batch, time, channels].
        wav_lens : torch.Tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.
        batch_id : torch.Tensor
            id of each batch (file names etc.)

        Returns
        -------
        list of dictionary: List[Dict[List]]
            The emotions and their temporal boundaries.
        """
        outputs = self.encode_batch(wavs, wav_lens)
        averaged_out = self.hparams.avg_pool(outputs)
        outputs = self.mods.output_mlp(averaged_out)
        outputs = self.hparams.log_softmax(outputs)
        score, index = torch.max(outputs, dim=-1)
        preds = self.hparams.label_encoder.decode_torch(index)
        results = self.preds_to_diarization(preds, batch_id)
        return results

    def preds_to_diarization(self, prediction, batch_id):
        """Convert frame-wise predictions into a dictionary of
        diarization results.

        Arguments
        ---------
        prediction : torch.Tensor
            Frame-wise predictions
        batch_id : str
            The id for this batch

        Returns
        -------
        dictionary
            A dictionary with the start/end of each emotion
        """
        results = {}

        for i in range(len(prediction)):
            pred = prediction[i]
            lol = []
            for j in range(len(pred)):
                start = round(self.hparams.stride * 0.02 * j, 2)
                end = round(start + self.hparams.window_length * 0.02, 2)
                lol.append([batch_id[i], start, end, pred[j]])

            lol = self.merge_ssegs_same_emotion_adjacent(lol)
            results[batch_id[i]] = [
                {"start": k[1], "end": k[2], "emotion": k[3]} for k in lol
            ]
        return results

    def forward(self, wavs, wav_lens, batch_id):
        """Get emotion diarization for a batch of waveforms."""
        return self.diarize_batch(wavs, wav_lens, batch_id)

    def is_overlapped(self, end1, start2):
        """Returns True if segments are overlapping.

        Arguments
        ---------
        end1 : float
            End time of the first segment.
        start2 : float
            Start time of the second segment.

        Returns
        -------
        overlapped : bool
            True of segments overlapped else False.

        Example
        -------
        >>> Speech_Emotion_Diarization.is_overlapped(None, 5.5, 3.4)
        True
        >>> Speech_Emotion_Diarization.is_overlapped(None, 5.5, 6.4)
        False
        """

        return start2 <= end1

    def merge_ssegs_same_emotion_adjacent(self, lol):
        """Merge adjacent sub-segs if they are the same emotion.

        Arguments
        ---------
        lol : list of list
            Each list contains [utt_id, sseg_start, sseg_end, emo_label].

        Returns
        -------
        new_lol : list of list
            new_lol contains adjacent segments merged from the same emotion ID.

        Example
        -------
        >>> from speechbrain.utils.EDER import merge_ssegs_same_emotion_adjacent
        >>> lol = [
        ...     ["u1", 0.0, 7.0, "a"],
        ...     ["u1", 7.0, 9.0, "a"],
        ...     ["u1", 9.0, 11.0, "n"],
        ...     ["u1", 11.0, 13.0, "n"],
        ...     ["u1", 13.0, 15.0, "n"],
        ...     ["u1", 15.0, 16.0, "a"],
        ... ]
        >>> merge_ssegs_same_emotion_adjacent(lol)
        [['u1', 0.0, 9.0, 'a'], ['u1', 9.0, 15.0, 'n'], ['u1', 15.0, 16.0, 'a']]
        """
        new_lol = []

        # Start from the first sub-seg
        sseg = lol[0]
        flag = False
        for i in range(1, len(lol)):
            next_sseg = lol[i]
            # IF sub-segments overlap AND has same emotion THEN merge
            if (
                self.is_overlapped(sseg[2], next_sseg[1])
                and sseg[3] == next_sseg[3]
            ):
                sseg[2] = next_sseg[2]  # just update the end time
                # This is important. For the last sseg, if it is the same emotion then merge
                # Make sure we don't append the last segment once more. Hence, set FLAG=True
                if i == len(lol) - 1:
                    flag = True
                    new_lol.append(sseg)
            else:
                new_lol.append(sseg)
                sseg = next_sseg
        # Add last segment only when it was skipped earlier.
        if flag is False:
            new_lol.append(lol[-1])
        return new_lol
