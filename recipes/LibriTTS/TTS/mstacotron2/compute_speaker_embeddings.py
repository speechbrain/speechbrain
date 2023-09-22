import json
from speechbrain.pretrained import EncoderClassifier, MelSpectrogramEncoder
import torchaudio
import pickle
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_speaker_embeddings(
    input_filepaths,
    output_file_paths,
    data_folder,
    spk_emb_encoder_path,
    spk_emb_sr,
    mel_spec_params,
    device,
):
    """This function processes a JSON file to compute the speaker embeddings

    Arguments
    ---------
    input_filepaths : list
        A list of paths to the JSON files to be processed
    output_file_paths : list
        A list of paths to the output pickle files corresponding to the input JSON files
    data_folder : str
        Path to the folder where LibriTTS data is stored
    spk_emb_encoder_path : str
        Path for the speaker encoder
    spk_emb_sr : int
        Sample rate used by the speaker embedding encoder
    mel_spec_params: dict
        Information about mel-spectrogram computation
    device : str
        Device for to be used for computation
    """

    # Checks if this phase is already done (if so, skips it)
    if skip(output_file_paths):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Initializes the speaker encoder
    spk_emb_encoder = None
    if mel_spec_params["custom_mel_spec_encoder"]:
        # To use the custom mel-spectrogram based encoder - for compatibility with future speaker consistency loss work
        spk_emb_encoder = MelSpectrogramEncoder.from_hparams(
            source=spk_emb_encoder_path, run_opts={"device": device}
        )
    else:
        # To use the speaker encoders available with SpeechBrain
        spk_emb_encoder = EncoderClassifier.from_hparams(
            source=spk_emb_encoder_path, run_opts={"device": device}
        )

    # Processes data manifests files to create corresponding speaker embedding files
    for i in range(len(input_filepaths)):
        logger.info(f"Creating {output_file_paths[i]}.")

        speaker_embeddings = dict()  # Holds speaker embeddings

        json_file = open(input_filepaths[i])
        json_data = json.load(json_file)

        # Processes all utterances in the data manifest file
        for utt_id, utt_data in tqdm(json_data.items()):
            utt_wav_path = utt_data["wav"]
            utt_wav_path = utt_wav_path.replace("{data_root}", data_folder)

            # Loads and resamples waveforms if required
            signal, sig_sr = torchaudio.load(utt_wav_path)
            if sig_sr != spk_emb_sr:
                signal = torchaudio.functional.resample(
                    signal, sig_sr, spk_emb_sr
                )
            signal = signal.to(device)

            # Computes the speaker embedding
            if mel_spec_params["custom_mel_spec_encoder"]:
                spk_emb = spk_emb_encoder.encode_waveform(signal)
            else:
                spk_emb = spk_emb_encoder.encode_batch(signal)

            spk_emb = spk_emb.squeeze()
            spk_emb = spk_emb.detach()

            speaker_embeddings[utt_id] = spk_emb.cpu()

        # Stores the speaker embeddings at the destination
        with open(output_file_paths[i], "wb") as output_file:
            pickle.dump(
                speaker_embeddings,
                output_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(f"Created {output_file_paths[i]}.")


def skip(filepaths):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filepath in filepaths:
        if not os.path.isfile(filepath):
            return False
    return True
