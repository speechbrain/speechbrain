def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


collect_ignore = ["setup.py"]
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/loss/transducer_loss.py")
try:
    import kenlm  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/decoders/language_model.py")
try:
    import fairseq  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/fairseq_wav2vec.py")
try:
    from transformers import Wav2Vec2Model  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append(
        "speechbrain/lobes/models/huggingface_transformers/wav2vec2.py"
    )
try:
    from transformers import WhisperModel  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append(
        "speechbrain/lobes/models/huggingface_transformers/whisper.py"
    )
try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/kmeans.py")
    collect_ignore.append(
        "speechbrain/lobes/models/huggingface_transformers/discrete_ssl.py"
    )
try:
    import peft  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append(
        "speechbrain/lobes/models/huggingface_transformers/llama2.py"
    )
try:
    import sacrebleu  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/bleu.py")
try:
    import vocos  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append(
        "speechbrain/lobes/models/huggingface_transformers/vocos.py"
    )
try:
    from speechtokenizer import SpeechTokenizer  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append(
        "speechbrain/lobes/models/discrete/speechTokenizer.py"
    )
try:
    import wavtokenizer  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/discrete/wavTokenzier.py")

try:
    from omegaconf import OmegaConf  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/discrete/SQ-Codec.py")
