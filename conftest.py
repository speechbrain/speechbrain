def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


collect_ignore = [
    "speechbrain/integrations/",
    # These can be removed once the modules are fully deprecated
    "speechbrain/utils/bleu.py",
    "speechbrain/utils/kmeans.py",
    "speechbrain/processing/diarization.py",
    "speechbrain/decoders/language_model.py",
    "speechbrain/alignment/ctc_segmentation.py",
    "speechbrain/lobes/models/fairseq_wav2vec.py",
    "speechbrain/lobes/models/kmeans.py",
]
