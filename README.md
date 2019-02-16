# PyTorch Layer Normalization

[![Travis](https://travis-ci.org/CyberZHG/torch-layer-normalization.svg)](https://travis-ci.org/CyberZHG/torch-layer-normalization)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/torch-layer-normalization/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/torch-layer-normalization)

Implementation of the paper: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

## Install

```bash
pip install torch-layer-normalization
```

## Usage

```python
from torch_layer_normalization import LayerNormalization

LayerNormalization(normal_shape=normal_shape)
# The `normal_shape` could be the last dimension of the input tensor or the shape of the input tensor.
```
