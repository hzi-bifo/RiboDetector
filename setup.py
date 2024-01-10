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
    "onnxruntime >= 1.10.0, <= 1.15.1",
    "torch >= 1.7.1, <= 1.12.1",
]

setup(
    name="ribodetector",
    version="0.3.1",
    python_requires=">=3.8, <=3.12",
    author="Z-L Deng",
    author_email="dawnmsg@gmail.com",
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
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'ribodetector = ribodetector.detect:main',
            'ribodetector_cpu = ribodetector.detect_cpu:main',
        ]
    },
    zip_safe=True,
    install_requires=required
)
