import speechbrain as sb
from speechbrain.dataio.sampler import (
    ConcatDatasetBatchSampler,
    ReproducibleRandomSampler,
)
from prepare_dcase2019_task4 import prepare_dcase2019_task4
from utils_dcase2019 import extract_ground_truth_list, ManyHotEncoder, signal
import math
import torch


def dataio_prep(hparams):
    """
    This Function call the tsv to json transform function and creates
    the data-pipeline and label-pipeline.Data-pipeline takes filepath
    and provides signal or read_audio.Label-pipeline takes filepath,
    event_label for weak samples and event_labels for strong samples.
    It provides the filename, strong and weak encoded labels, strong
    truth labels.Dynamic Item dataset is created for each type of the data,
    strong_train, weak_train, strong+weak_train,validation, test dataset.
    Arguments
        ---------
        hparams : .yaml
            yaml file with all configuration

    Returns
    -------
        tuple (5) of Dict of Dynamic Item Datasets and batch samplers
            ie (dataset, batch_sampler_1, batch_sampler_2, batch_sampler_3, batch_sampler_4)
    """

    ###################################################
    ## Creates the required json file  from the tsv  ##
    ###################################################

    prepare_dcase2019_task4(
        hparams["JsonOutput"],
        hparams["AudioPath"],
        hparams["MetaDataPath"],
        hparams["AudioPath_Unlabel"],
        hparams["MissingFilesPath"],
    )

    # actual data pipeline and label pipeline
    max_frames = math.ceil(
        hparams["max_len_seconds"]
        * hparams["sample_rate"]
        / hparams["hop_length"]
    )
    max_length = hparams["max_len_seconds"] * hparams["sample_rate"]

    ###############################
    ##  Data Pipeline creation   ##
    ###############################
    @sb.utils.data_pipeline.takes("filepath")
    @sb.utils.data_pipeline.provides("sig", "mfcc")
    def data_pipeline(filepath):
        sig = signal(filepath, max_length)
        yield sig

    ###############################
    ## Label Pipeline creation  ###
    ###############################
    @sb.utils.data_pipeline.takes("filepath", "event_label", "event_labels")
    @sb.utils.data_pipeline.provides(
        "filename", "strong_truth", "strong_encoded", "weak_encoded"
    )
    def label_pipeline(filepath, event_label, event_labels):
        filename = filepath.split("/")[-1]
        yield filename
        strong_truth = extract_ground_truth_list(event_label, filename)
        yield strong_truth
        mhe = ManyHotEncoder(
            hparams["classes"],
            hparams["max_len_seconds"],
            max_frames // hparams["pooling_time_ratio"],
        )
        strong_encoded = mhe.encode_strong(event_label)
        yield strong_encoded
        weak_encoded = mhe.encode_weak(event_labels)
        yield weak_encoded

    ##############################################################
    ##  Creating Dict of Dynamic Item datasets + batch sampler  ##
    ##############################################################
    datasets = {}
    datasets["train_weak"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["train"]["weak"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets[
        "train_synthetic"
    ] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["train"]["synthetic"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets["train_unlabel"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["train"]["unlabel_in_domain"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets["train_weak_synthetic"] = torch.utils.data.ConcatDataset(
        [datasets["train_weak"], datasets["train_synthetic"]]
    )

    datasets["train_weak_synthetic_unlabel"] = torch.utils.data.ConcatDataset(
        [
            datasets["train_weak"],
            datasets["train_unlabel"],
            datasets["train_synthetic"],
        ]
    )

    sampler1 = ReproducibleRandomSampler(datasets["train_weak"])
    sampler2 = ReproducibleRandomSampler(datasets["train_unlabel"])
    sampler3 = ReproducibleRandomSampler(datasets["train_synthetic"])

    batch_sampler_2 = ConcatDatasetBatchSampler(
        [sampler1, sampler3],
        [hparams["batch_size"] // 2, hparams["batch_size"] // 2],
    )
    batch_sampler_3 = ConcatDatasetBatchSampler(
        [sampler1, sampler2, sampler3],
        [
            hparams["batch_size"] // 6,
            2 * hparams["batch_size"] // 3,
            hparams["batch_size"] // 6,
        ],
    )

    datasets["validation"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["validation"]["validation"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets["test"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["eval"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    ### toy datsets
    datasets["train_weak_toy"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["toy"]["weak"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets[
        "train_synthetic_toy"
    ] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["toy"]["synthetic"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets[
        "train_unlabel_toy"
    ] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["toy"]["unlabel"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets["train_weak_synthetic_toy"] = torch.utils.data.ConcatDataset(
        [datasets["train_weak_toy"], datasets["train_synthetic_toy"]]
    )

    datasets[
        "train_weak_synthetic_unlabel_toy"
    ] = torch.utils.data.ConcatDataset(
        [
            datasets["train_weak_toy"],
            datasets["train_unlabel_toy"],
            datasets["train_synthetic_toy"],
        ]
    )

    sampler1_toy = ReproducibleRandomSampler(datasets["train_weak_toy"])
    sampler2_toy = ReproducibleRandomSampler(datasets["train_unlabel_toy"])
    sampler3_toy = ReproducibleRandomSampler(datasets["train_synthetic_toy"])

    batch_sampler_2_toy = ConcatDatasetBatchSampler(
        [sampler1, sampler3],
        [hparams["batch_size"] // 2, hparams["batch_size"] // 2],
    )

    batch_sampler_3_toy = ConcatDatasetBatchSampler(
        [sampler1_toy, sampler2_toy, sampler3_toy],
        [
            hparams["batch_size"] // 4,
            hparams["batch_size"] // 2,
            hparams["batch_size"] // 4,
        ],
    )

    datasets["validation_toy"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["toy"]["validation"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    datasets["test_toy"] = sb.dataio.dataset.DynamicItemDataset.from_json(
        hparams["JsonMetaData"]["toy"]["test"],
        dynamic_items=[data_pipeline, label_pipeline],
        output_keys=[
            "sig",
            "filename",
            "strong_truth",
            "strong_encoded",
            "weak_encoded",
        ],
    )

    return (
        datasets,
        batch_sampler_2,
        batch_sampler_3,
        batch_sampler_2_toy,
        batch_sampler_3_toy,
    )
