"""Downloads or otherwise fetches pretrained models"""
import huggingface_hub
import re
import os


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


def _fetch_all_from_hub(model_url, local_dir, **huggingface_kwargs):
    huggingface_hub.Repository(
        local_dir, clone_from=model_url, **huggingface_kwargs
    )


def fetch(
    hub_modelID,
    local_dir,
    hparams_file="hyperparams.yaml",
    **huggingface_kwargs,
):
    """
    """
    if os.path.exists(local_dir) and os.path.join(local_dir, hparams_file):
        print("Model already exists locally, loading local copy")
    else:
        _fetch_all_from_hub(hub_modelID, local_dir, **huggingface_kwargs)
    return os.path.join(local_dir, hparams_file)
