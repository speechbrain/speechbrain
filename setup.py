#!/usr/bin/env python3
import os
import site
import sys
from distutils.core import setup

import setuptools

# Editable install in user site directory can be allowed with this hack:
# https://github.com/pypa/pip/issues/7953.
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join("speechbrain", "version.txt"), encoding="utf-8") as f:
    version = f.read().strip()

setup(
    name="speechbrain",
    version=version,
    description="All-in-one speech toolkit in pure Python and Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mirco Ravanelli, Titouan Parcollet & Others",
    author_email="speechbrain@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    # we don't want to ship the tests package. for future proofing, also
    # exclude any tests subpackage (if we ever define __init__.py there)
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    package_data={"speechbrain": ["version.txt", "log-config.yaml"]},
    install_requires=[
        "hyperpyyaml",
        "joblib",
        "numpy",
        "packaging",
        "scipy",
        "sentencepiece",
        "torch>=1.9",
        "torchaudio",
        "tqdm",
        "huggingface_hub",
    ],
    python_requires=">=3.8",
    url="https://speechbrain.github.io/",
)
