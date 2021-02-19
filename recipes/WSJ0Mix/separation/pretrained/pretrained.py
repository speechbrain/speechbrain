"""
Ready to use (pretrained) speech separation models
Authors
 * Cem Subakan 2021
 * Mirco Ravanelli 2021
"""

import os
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import download_file
from recipes.WSJ0Mix.separation.train.train import Separation


class pretrained_separator(Separation):
    """A "ready-to-use" speech separation model for 2 or 3 speaker mixtures.
    This pretrained model is based on the WSJ0Mix recipe.

    The SepFormer system achieves an SI-SNR=22.3dB on WSJ0-2Mix test set.

    Arguments
    ---------
    hparams_file : str
        Path where the yaml file with the model definition is stored.
        If it is an url, the yaml file is downloaded.
    encoder_file : str
        Path where the parameters of the encoder (.ckpt file) is stored.
        If it is an url, the checkpoint file is downloaded.
    mask_file : str
        Path where the parameters of the masknet (.ckpt file) is stored.
        If it is an url, the checkpoint file is downloaded.
    decoder_file : str
        Path where the parameters of the decoder (.ckpt file) is stored.
        If it is an url, the checkpoint file is downloaded.
    save_folder : str
        Path where the lm (yaml + model) will be saved (default 'separation_model')

    Example
    -------
    >>> import torch
    >>> model = pretrained_separator()
    Downloading https://drive.google.com/uc?export=download&id=1AHbegv1btiINzmXJsZyZ9qQL-e-6f8hH to separation_model/sepformer_wsj02mix_dm.yaml
    Downloading https://drive.google.com/uc?export=download&id=1OQcKtJjF5bn9I7WQ3uMY-0RBD9DW5nto to separation_model/encoder.ckpt
    Downloading https://www.dropbox.com/s/jqjy6pr9rz0lwda/masknet.ckpt?dl=1 to separation_model/masknet.ckpt
    Downloading https://drive.google.com/uc?export=download&id=1R0AsN9f-5aYB-aeKyL2o5xNKeKFmBMC2 to separation_model/decoder.ckpt
    >>> mix = torch.randn(1, 160)
    >>> result, _ = model.compute_forward([mix, torch.tensor([1])], [[mix], [mix]], stage='test')
    >>> print(result.shape)
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        hparams_file="https://drive.google.com/uc?export=download&id=1AHbegv1btiINzmXJsZyZ9qQL-e-6f8hH",
        encoder_file="https://drive.google.com/uc?export=download&id=1OQcKtJjF5bn9I7WQ3uMY-0RBD9DW5nto",
        masknet_file="https://www.dropbox.com/s/jqjy6pr9rz0lwda/masknet.ckpt?dl=1",
        decoder_file="https://drive.google.com/uc?export=download&id=1R0AsN9f-5aYB-aeKyL2o5xNKeKFmBMC2",
        save_folder="separation_model",
        overrides={},
    ):
        """Downloads the pretrained modules specified in the yaml"""

        self.encoder_file = encoder_file
        self.masknet_file = masknet_file
        self.decoder_file = decoder_file
        save_model_path = os.path.join(
            save_folder, "sepformer_wsj02mix_dm.yaml"
        )
        download_file(hparams_file, save_model_path)
        hparams_file = save_model_path

        # Loading modules defined in the yaml file
        with open(hparams_file) as fin:
            overrides["save_folder"] = save_folder
            self.hyparams = load_hyperpyyaml(fin, overrides)

        # initialize the inherited class
        super(pretrained_separator, self).__init__(hparams=self.hyparams)

        self.device = "cuda" if torch.cuda.is_available else "cpu"

        # Creating directory where pre-trained models are stored
        if not os.path.isdir(self.hyparams["save_folder"]):
            os.makedirs(self.hyparams["save_folder"])

        # putting modules on the right device
        self.mod = torch.nn.ModuleDict(self.hyparams["modules"]).to(self.device)

        # Load pretrained modules
        self.load_separator()

    def load_separator(self):
        """Loads the separator model specified in the yaml file"""
        encoder_savepath = os.path.join(
            self.hyparams["save_folder"], "encoder.ckpt"
        )
        masknet_savepath = os.path.join(
            self.hyparams["save_folder"], "masknet.ckpt"
        )
        decoder_savepath = os.path.join(
            self.hyparams["save_folder"], "decoder.ckpt"
        )

        download_file(self.encoder_file, encoder_savepath)
        download_file(self.masknet_file, masknet_savepath)
        download_file(self.decoder_file, decoder_savepath)

        self.mod.encoder.load_state_dict(
            torch.load(encoder_savepath, map_location=self.device), strict=True
        )
        self.mod.masknet.load_state_dict(
            torch.load(masknet_savepath, map_location=self.device), strict=True
        )
        self.mod.decoder.load_state_dict(
            torch.load(decoder_savepath, map_location=self.device), strict=True
        )


if __name__ == "__main__":
    # an example separation with an arbitrary mixture

    # instantiate the pretrained model
    model = pretrained_separator()

    # load the audio
    audio_file1 = "../../../../samples/audio_samples/example1.wav"
    audio_file2 = "../../../../samples/audio_samples/example2.flac"
    wav1, fs = torchaudio.load(audio_file1)
    wav2, fs2 = torchaudio.load(audio_file2)

    # subsample to 8k
    wav1, wav2 = wav1[:, ::2], wav2[:, ::2]

    # form the mixture
    min_len = min(wav1.shape[1], wav2.shape[1])
    mix = 2 * wav1[:, :min_len] + wav2[:, :min_len]
    mix = mix.to(model.device)

    # forward pass
    result, _ = model.compute_forward(
        [mix, torch.tensor([1])], [[mix], [mix]], stage="test"
    )

    # normalize
    result = result / result.max(dim=1, keepdim=True)[0]

    # save the results and the mixture
    source1hat = result[:, :, 0]
    source2hat = result[:, :, 1]
    torchaudio.save("mix.wav", mix.cpu(), 8000)
    torchaudio.save("source1hat.wav", result[:, :, 0].detach().cpu(), 8000)
    torchaudio.save("source2hat.wav", result[:, :, 1].detach().cpu(), 8000)
