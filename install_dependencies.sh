#!/bin/bash

# Install wheel
pip install wheel

# Install specific versions of torch, torchvision, and torchaudio with CPU support
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install additional Python packages
pip install torchdata==0.6.0 numpy==1.26.4 pandas dgl seaborn matplotlib networkx torch_geometric pyarrow tqdm ogb pyyaml pydantic torch_scatter torch_sparse

# Download graphlearn_torch repository
git clone https://github.com/alibaba/graphlearn-for-pytorch.git

# Change directory to graphlearn-for-pytorch
cd graphlearn-for-pytorch

# Build graphlearn_torch
WITH_CUDA=OFF python setup.py bdist_wheel
pip install dist/*

# Install Pip wheel
pip install graphlearn-torch

# Change directory to reorder-training
cd ../training
