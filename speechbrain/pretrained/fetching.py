"""Downloads or otherwise fetches pretrained models"""
import huggingface_hub
import re
import os
import logging

logger = logging.getLogger(__name__)


def list_models(user="sb", regex=None):

    regex = re.compile(regex)
    api = huggingface_hub.HfApi()
    models = []
    for m in api.model_list():
        if m.modelId.startswith(user):
            if regex is not None:
                if re.match(regex, m.modelId):
                    models.append(m)
            else:
                models.append(m)
    return models


def fetch(local_path, filename, hub_modelID=None, **download_kwargs):

    local_path = os.path.abspath(local_path)
    local_abs_path = os.path.join(local_path, filename)
    if os.path.isfile(local_abs_path):
        logger.info(
            "Requested file {} exists locally, using local copy.".format(
                filename
            )
        )
    else:
        if hub_modelID is None:
            logger.error(
                "Requested file {} does not exists locally. It can be downloaded from HuggingFace ModelHub using 'hub_modelID' argument".format(
                    filename
                )
            )
            raise FileNotFoundError

        os.makedirs(local_path)

        url = huggingface_hub.hf_hub_url(hub_modelID, filename)
        logger.info(
            "Downloading requested file {} from {}.".format(filename, url)
        )

        fetched_file = huggingface_hub.cached_download(
            url, cache_dir=local_path, **download_kwargs
        )

        os.rename(os.path.join(local_path, fetched_file), local_abs_path)
    return local_abs_path
