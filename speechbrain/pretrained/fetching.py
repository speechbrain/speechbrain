"""Downloads or otherwise fetches pretrained models"""
import huggingface_hub


def fetch(filename, source):
    """Fetch a file from Huggingface hub / filesystem / other
    """
    # In the filesystem the full path would be source/file
    # If source exists in system, copy or maybe symlink here.
    # Otherwise look in huggingface hub and cached_download here.
    huggingface_hub
    pass
