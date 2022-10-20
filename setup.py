#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="HSE NLP course homework",
    author="Artem Makoyan",
    author_email="makoyan2001@gmail.com",
    url="https://github.com/MakArtKar/cinderella_telegram",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
