#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import find_packages, setup

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

required = [
    "pandas",
    "tqdm",
    "numpy",
    "biopython",
    "onnxruntime",
    "torch == 1.7.1",
]

setup(
    name="ribodetector",
    version="0.2.2",
    author="Z-L Deng",
    author_email="dawnmsg(at)gmail.com",
    description="Accurate and rapid RiboRNA sequences Detector based on deep learning.",
    license="GPL-3 License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hzi-bifo/RiboDetector",
    packages=find_packages(include=["ribodetector", "ribodetector.*"]),
    package_data={'': ['*.json', '*.yaml', '*.pth', '*.onnx']},

    include_package_data=True,

    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bioinformatics",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3 License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'ribodetector = ribodetector.detect:main',
            'ribodetector_cpu = ribodetector.detect_cpu:main',
        ]
    },
    zip_safe=False,
    install_requires=required
)