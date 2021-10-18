#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Copyright (C) 2021 The BiometricBlender contributors.
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biometricblender",
    version="0.0.1",
    author="Marcell Stippinger",
    author_email="stippingerm.prog@gmail.com",
    description="Ultra-high dimensional,multi-class synthetic data generator to imitatebiometric feature space",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cursorinsight/biometricblender",
    project_urls={
        "Bug Tracker": "https://github.com/cursorinsight/biometricblender/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7.1",
    install_requires=['h5py>=2.10',
                      'numpy>=1.18',
                      'scipy>=1.6',
                      'scikit-learn>=0.24']
)
