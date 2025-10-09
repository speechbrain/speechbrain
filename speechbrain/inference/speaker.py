"""Specifies the inference interfaces for speaker recognition modules.

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

from speechbrain.inference.classifiers import EncoderClassifier


class SpeakerRecognition(EncoderClassifier):
    """A ready-to-use model for speaker recognition. It can be used to
    perform speaker verification with verify_batch().

    Arguments
    ---------
    *args : tuple
    **kwargs : dict
        Arguments are forwarded to ``Pretrained`` parent class.

    Example
    -------
    >>> import torchaudio
    >>> from speechbrain.inference.speaker import SpeakerRecognition
    >>> # Model is downloaded from the speechbrain HuggingFace repo
    >>> tmpdir = getfixture("tmpdir")
    >>> verification = SpeakerRecognition.from_hparams(
    ...     source="speechbrain/spkrec-ecapa-voxceleb",
    ...     savedir=tmpdir,
    ... )

    >>> # Perform verification
    >>> signal, fs = torchaudio.load("tests/samples/single-mic/example1.wav")
    >>> signal2, fs = torchaudio.load("tests/samples/single-mic/example2.flac")
    >>> score, prediction = verification.verify_batch(signal, signal2)
    """

    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model",
        "mean_var_norm_emb",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def verify_batch(
        self, wavs1, wavs2, wav1_lens=None, wav2_lens=None, threshold=0.25
    ):
        """Performs speaker verification with cosine distance.

        It returns the score and the decision (0 different speakers,
        1 same speakers).

        Arguments
        ---------
        wavs1 : Torch.Tensor
            torch.Tensor containing the speech waveform1 (batch, time).
            Make sure the sample rate is fs=16000 Hz.
        wavs2 : Torch.Tensor
            torch.Tensor containing the speech waveform2 (batch, time).
            Make sure the sample rate is fs=16000 Hz.
        wav1_lens : Torch.Tensor
            torch.Tensor containing the relative length for each sentence
            in the length (e.g., [0.8 0.6 1.0])
        wav2_lens : Torch.Tensor
            torch.Tensor containing the relative length for each sentence
            in the length (e.g., [0.8 0.6 1.0])
        threshold : Float
            Threshold applied to the cosine distance to decide if the
            speaker is different (0) or the same (1).

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        emb1 = self.encode_batch(wavs1, wav1_lens, normalize=False)
        emb2 = self.encode_batch(wavs2, wav2_lens, normalize=False)
        score = self.similarity(emb1, emb2)
        return score, score > threshold

    def verify_files(self, path_x, path_y, **kwargs):
        """Speaker verification with cosine distance

        Returns the score and the decision (0 different speakers,
        1 same speakers).

        Arguments
        ---------
        path_x : str
            Path to file x
        path_y : str
            Path to file y
        **kwargs : dict
            Arguments to ``load_audio``

        Returns
        -------
        score
            The score associated to the binary verification output
            (cosine distance).
        prediction
            The prediction is 1 if the two signals in input are from the same
            speaker and 0 otherwise.
        """
        waveform_x = self.load_audio(path_x, **kwargs)
        waveform_y = self.load_audio(path_y, **kwargs)
        # Fake batches:
        batch_x = waveform_x.unsqueeze(0)
        batch_y = waveform_y.unsqueeze(0)
        # Verify:
        score, decision = self.verify_batch(batch_x, batch_y)
        # Squeeze:
        return score[0], decision[0]
