---
hide:
- navigation
---
# OD-Metrics
<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/images/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/images/logo_light.svg">
  <img width="400" height="400" src="assets/images/logo_dark.svg">
</picture>
</div>
<p align="center">
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/EMalagoli92/3f159a4246243b883a5c817ca2d34baa/raw/unit_test.json?kill_cache=1" />
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/EMalagoli92/d23fd688b541d4b303d2baa6ee87e51a/raw/mypy.json?kill_cache=1" />
  <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/EMalagoli92/3ab4a977b9a0e4ccb7178dd1fa51e1b0/raw/pylint.json?kill_cache=1" />
  <a href="https://codecov.io/gh/EMalagoli92/OD-Metrics">
    <img src="https://codecov.io/gh/EMalagoli92/OD-Metrics/graph/badge.svg?token=U7VJTKGYN6"></a>
  <a href="https://pypi.org/project/od-metrics/#description">
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/EMalagoli92/331395960725a4b47d4ca4977a24e949/raw/version.json?kill_cache=1"></a>
  <br>
  <img src="https://img.shields.io/badge/python-%3E=3.9-yellow.svg?style=flat">
  <a href="https://github.com/EMalagoli92/OD-Metrics/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat" alt="License: MIT"></a><br>
  <a href="https://mybinder.org/v2/gh/EMalagoli92/OD-metrics/HEAD?labpath=samples%2Fsamples.ipynb">
    <img src="https://mybinder.org/badge_logo.svg"></a>
  <a href="https://colab.research.google.com/github/EMalagoli92/OD-Metrics/blob/main/samples/samples.ipynb">
    <img src="https://img.shields.io/badge/Open%20in%20Colab-blue?logo=google-colab&style=flat&labelColor=555"></a>
</p>

<p align="center">
  <strong>
    A python library for Object Detection metrics.
  </strong>
</p>


## Why OD-Metrics?
- **User-friendly**: Designed for simplicity, allowing users to calculate metrics with minimal setup.
- **Highly Customizable**: Offers flexibility by allowing users to set custom values for every parameter in metrics definitions.
- **Compatibility with COCOAPI**: Metrics are rigorously tested to ensure compatibility with [COCOAPI](https://github.com/cocodataset/cocoapi), ensuring reliability and consistency.

## Metrics Overview
Supported metrics include **[mAP](map.md) (Mean Average Precision)**, `mAR` (Mean Average Recall)
and **[IoU](iou.md) (Intersection over Union)**. For more information see [Metrics](iou.md).


## Installation
Install from PyPI
```
pip install od-metrics
```
Install from Github
```
pip install git+https://github.com/EMalagoli92/OD-Metrics
```

## License

This work is made available under the [MIT License](https://github.com/EMalagoli92/OD-Metrics/blob/main/LICENSE)
