"""
Managing the logger.
"""

import os
import yaml
import logging
import logging.config
from speechbrain.utils.data_utils import recursive_update


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration

    Args:
        config_path: the path to a logging config file.
        default_level: the level to use if config file is not found.
        overrides: a dictionary of the same structure as the config dict
            with any updated values that need to be applied.

    Author:
        Fang-Pen Lin 2012 and Peter Plantinga 2020
        https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
