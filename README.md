<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/EMalagoli92/OD-Metrics/main/docs/assets/images/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/EMalagoli92/OD-Metrics/main/docs/assets/images/logo_light.svg">
  <img width="400" height="400" src="https://raw.githubusercontent.com/EMalagoli92/OD-Metrics/main/docs/assets/images/logo_light.svg">
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
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat" alt="License: Apache 2.0"></a><br>
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
- **COCOAPI Compatibility**: Metrics are rigorously tested to ensure compatibility with [COCOAPI](https://github.com/cocodataset/cocoapi), ensuring reliability and consistency.


## Supported Metrics
Supported metrics include:

- [mAP](https://emalagoli92.github.io/OD-Metrics/map_mar/) (Mean Average Precision)
- [mAR](https://emalagoli92.github.io/OD-Metrics/map_mar/#average-recall) (Mean Average Recall)
- [IoU](https://emalagoli92.github.io/OD-Metrics/iou/) (Intersection over Union).

For more information see [Metrics](https://emalagoli92.github.io/OD-Metrics/iou/) documentation.

## Documentation
For help, usage, API reference, and an overview of metrics formulas, please refer to [Documentation](https://emalagoli92.github.io/OD-Metrics/).


## Try live Demo
Try `OD-Metrics` samples [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMalagoli92/OD-metrics/HEAD?labpath=samples%2Fsamples.ipynb)
  <a href="https://colab.research.google.com/github/EMalagoli92/OD-Metrics/blob/main/samples/samples.ipynb">
    <img src="https://img.shields.io/badge/Open%20in%20Colab-blue?logo=google-colab&style=flat&labelColor=555"></a>


## Installation
Install from PyPI
```
pip install od-metrics
```
Install from Github
```
pip install git+https://github.com/EMalagoli92/OD-Metrics
```


## Simple Example

``` python
from od_metrics import ODMetrics

# Ground truths
y_true = [
    { # image 1
     "boxes": [[25, 16, 38, 56], [129, 123, 41, 62]],
     "labels": [0, 1]
     },
    { # image 2
     "boxes": [[123, 11, 43, 55], [38, 132, 59, 45]],
     "labels": [0, 0]
     }
    ]

# Predictions
y_pred = [
    { # image 1
     "boxes": [[25, 27, 37, 54], [119, 111, 40, 67], [124, 9, 49, 67]],
     "labels": [0, 1, 1],
     "scores": [.88, .70, .80]
     },
    { # image 2
     "boxes": [[64, 111, 64, 58], [26, 140, 60, 47], [19, 18, 43, 35]],
     "labels": [0, 1, 0],
     "scores": [.71, .54, .74]
     }
    ]

metrics = ODMetrics()
output = metrics.compute(y_true, y_pred)
print(output)
"""
{
    "mAP@[.5 | all | 100]": 0.16831683168316827,
    "mAP@[.5:.95 | all | 100]": 0.06732673267326732,
    "mAP@[.5:.95 | large | 100]": -1.0,
    "mAP@[.5:.95 | medium | 100]": 0.06732673267326732,
    "mAP@[.5:.95 | small | 100]": -1.0,
    "mAP@[.75 | all | 100]": 0.0,
    "mAR@[.5 | all | 100]": 0.16666666666666666,
    "mAR@[.5:.95 | all | 100]": 0.06666666666666667,
    "mAR@[.5:.95 | all | 10]": 0.06666666666666667,
    "mAR@[.5:.95 | all | 1]": 0.06666666666666667,
    "mAR@[.5:.95 | large | 100]": -1.0,
    "mAR@[.5:.95 | medium | 100]": 0.06666666666666667,
    "mAR@[.5:.95 | small | 100]": -1.0,
    "mAR@[.75 | all | 100]": 0.0,
    "class_metrics": {
        "0": {
            "AP@[.5 | all | 100]": 0.33663366336633654,
            "AP@[.5:.95 | all | 100]": 0.13465346534653463,
            "AP@[.5:.95 | large | 100]": -1.0,
            "AP@[.5:.95 | medium | 100]": 0.13465346534653463,
            "AP@[.5:.95 | small | 100]": -1.0,
            "AP@[.75 | all | 100]": 0.0,
            "AR@[.5 | all | 100]": 0.3333333333333333,
            "AR@[.5:.95 | all | 100]": 0.13333333333333333,
            "AR@[.5:.95 | all | 10]": 0.13333333333333333,
            "AR@[.5:.95 | all | 1]": 0.13333333333333333,
            "AR@[.5:.95 | large | 100]": -1.0,
            "AR@[.5:.95 | medium | 100]": 0.13333333333333333,
            "AR@[.5:.95 | small | 100]": -1.0,
            "AR@[.75 | all | 100]": 0.0
        },
        "1": {
            "AP@[.5 | all | 100]": 0.0,
            "AP@[.5:.95 | all | 100]": 0.0,
            "AP@[.5:.95 | large | 100]": -1.0,
            "AP@[.5:.95 | medium | 100]": 0.0,
            "AP@[.5:.95 | small | 100]": -1.0,
            "AP@[.75 | all | 100]": 0.0,
            "AR@[.5 | all | 100]": 0.0,
            "AR@[.5:.95 | all | 100]": 0.0,
            "AR@[.5:.95 | all | 10]": 0.0,
            "AR@[.5:.95 | all | 1]": 0.0,
            "AR@[.5:.95 | large | 100]": -1.0,
            "AR@[.5:.95 | medium | 100]": 0.0,
            "AR@[.5:.95 | small | 100]": -1.0,
            "AR@[.75 | all | 100]": 0.0
        }
    },
    "classes": [
        0,
        1
    ],
    "n_images": 2
}
"""
```


## Aknowledgment
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics)
- [COCO API](https://github.com/cocodataset/cocoapi)
- [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

## License
This work is made available under the [Apache License 2.0](https://github.com/EMalagoli92/OD-Metrics/blob/main/LICENSE)

## Support
Found this helpful? ⭐ it on [GitHub](https://github.com/EMalagoli92/OD-Metrics)
