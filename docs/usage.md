---
hide:
- navigation
---
## Try live Demo
Explore live `OD-Metrics` examples on Binder [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EMalagoli92/OD-metrics/HEAD?labpath=samples%2Fsamples.ipynb) or Google Colab <a href="https://colab.research.google.com/github/EMalagoli92/OD-Metrics/blob/main/samples/samples.ipynb">
    <img src="https://img.shields.io/badge/Open%20in%20Colab-blue?logo=google-colab&style=flat&labelColor=555"></a>

## Simple example.
Consider a scenario with two images, **Image 1** and **Image 2**, and the following annotations and predictions.

<img align="left" width="450" height="600" src="../assets/images/image_1.png">
**Image 1** contains:

- `2` ground-truth bounding boxes, one for class `0` and one for class `1`;
- `3` predicted bounding boxes with `labels` `[0, 1, 1]` and `scores` `[.88, .70, .80]`.
```yaml
# Image 1
y_true =
  {
   "boxes": [[25, 16, 38, 56], [129, 123, 41, 62]],
   "labels": [0, 1]
   }
y_pred =
  {
   "boxes": [[25, 27, 37, 54], [119, 111, 40, 67], [124, 9, 49, 67]],
   "labels": [0, 1, 1],
   "scores": [.88, .70, .80]
   },
```

<img align="left" width="450" height="600" src="../assets/images/image_2.png">
**Image 2** contains:

- `2` ground-truth bounding boxes, both for class `0`;
- `3` predicted bounding boxes, with `labels` `[0, 1, 0]` and `scores` `[.71, .54, .74]`.
```yaml
# Image 2
y_true =
  {
   "boxes": [[123, 11, 43, 55], [38, 132, 59, 45]],
   "labels": [0, 0]
   }
y_pred = {
   "boxes": [[64, 111, 64, 58], [26, 140, 60, 47], [19, 18, 43, 35]],
   "labels": [0, 1, 0],
   "scores": [.71, .54, .74]
   }
```
<br>

The [mAP](map_mar.md#what-is-map) (Mean Average Precision) and [mAR](map_mar.md#average-recall) (Mean Average Recall)
for this scenario are computed using `OD-Metrics` as follows.
``` py title="simple_example"
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


## Custom settings
By default, `OD-Metrics` follows [MS-COCO](https://cocodataset.org/#home) [@lin2014microsoft] settings, including `iou_thresholds`, `recall_thresholds`, `max_detection_thresholds`, `area_ranges`, and `class_metrics` (see [ODMetrics.\__init__()][src.od_metrics.od_metrics.ODMetrics.__init__] method).<br>
Custom settings can replace the default configuration. For instance, to set an IoU threshold of `0.4` and a maximum detection
threshold of `2`:

``` py title="custom_settings_example"
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

metrics = ODMetrics(iou_thresholds=.4, max_detection_thresholds=2)
output = metrics.compute(y_true, y_pred)
print(output)
"""
{
    "mAP@[.4 | all | 2]": 0.4183168316831683,
    "mAP@[.4 | large | 2]": -1.0,
    "mAP@[.4 | medium | 2]": 0.4183168316831683,
    "mAP@[.4 | small | 2]": -1.0,
    "mAR@[.4 | all | 2]": 0.6666666666666666,
    "mAR@[.4 | large | 2]": -1.0,
    "mAR@[.4 | medium | 2]": 0.6666666666666666,
    "mAR@[.4 | small | 2]": -1.0,
    "class_metrics": {
        "0": {
            "AP@[.4 | all | 2]": 0.33663366336633654,
            "AP@[.4 | large | 2]": -1.0,
            "AP@[.4 | medium | 2]": 0.33663366336633654,
            "AP@[.4 | small | 2]": -1.0,
            "AR@[.4 | all | 2]": 0.3333333333333333,
            "AR@[.4 | large | 2]": -1.0,
            "AR@[.4 | medium | 2]": 0.3333333333333333,
            "AR@[.4 | small | 2]": -1.0
        },
        "1": {
            "AP@[.4 | all | 2]": 0.5,
            "AP@[.4 | large | 2]": -1.0,
            "AP@[.4 | medium | 2]": 0.5,
            "AP@[.4 | small | 2]": -1.0,
            "AR@[.4 | all | 2]": 1.0,
            "AR@[.4 | large | 2]": -1.0,
            "AR@[.4 | medium | 2]": 1.0,
            "AR@[.4 | small | 2]": -1.0
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
## `class_metrics`
If `True`, evaluation is performed per class: detections are matched to ground truths only if they share the same `label_id`. If `False`, evaluation is category-agnostic. When `True`, the output includes a `"class_metrics"`
dictionary with per-class results. This corresponds to `useCats` in the COCO evaluation protocol. If not specified the default (COCO) is used and
corresponds to `True`.<br>
By setting `class_metrics=False`, the evaluation is category-agnostic.
``` py title="class_metrics_example"
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

metrics = ODMetrics(class_metrics=False)
output = metrics.compute(y_true, y_pred)
print(output)
"""
{
    "mAP@[.5 | all | 100]": 0.2574257425742574,
    "mAP@[.5:.95 | all | 100]": 0.10297029702970294,
    "mAP@[.5:.95 | large | 100]": -1.0,
    "mAP@[.5:.95 | medium | 100]": 0.10297029702970294,
    "mAP@[.5:.95 | small | 100]": -1.0,
    "mAP@[.75 | all | 100]": 0.0,
    "mAR@[.5 | all | 100]": 0.25,
    "mAR@[.5:.95 | all | 100]": 0.1,
    "mAR@[.5:.95 | all | 10]": 0.1,
    "mAR@[.5:.95 | all | 1]": 0.1,
    "mAR@[.5:.95 | large | 100]": -1.0,
    "mAR@[.5:.95 | medium | 100]": 0.1,
    "mAR@[.5:.95 | small | 100]": -1.0,
    "mAR@[.75 | all | 100]": 0.0,
    "classes": [
        0,
        1
    ],
    "n_images": 2
}
"""
```

## `extended_summary`
The `extended_summary` option in the [ODMetrics.compute()][src.od_metrics.od_metrics.ODMetrics.compute] method enables an extended summary with additional metrics such as `IoU`, `AP` (Average Precision), `AR` (Average Recall), and `mean_evaluator` (a `Callable`).

``` py title="extended_summary_example"
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
output = metrics.compute(y_true, y_pred, extended_summary=True)
print(list(output.keys()))
"""
['mAP@[.5 | all | 100]',,
 'mAP@[.5:.95 | all | 100]',
 'mAP@[.5:.95 | large | 100]',
 'mAP@[.5:.95 | medium | 100]',
 'mAP@[.5:.95 | small | 100]',
 'mAP@[.75 | all | 100]',
 'mAR@[.5 | all | 100]',
 'mAR@[.5:.95 | all | 100]',
 'mAR@[.5:.95 | all | 10]',
 'mAR@[.5:.95 | all | 1]',
 'mAR@[.5:.95 | large | 100]',
 'mAR@[.5:.95 | medium | 100]',
 'mAR@[.5:.95 | small | 100]',
 'mAR@[.75 | all | 100]',
 'classes',
 'n_images',
 'AP',
 'AR',
 'IoU',
 'mean_evaluator']
"""
```
In particular, `mean_evaluator` is a `Callable` that can calculate metrics for any combination of settings, even those not included in default `compute` output. For example, with standard [MS-COCO](https://cocodataset.org/#home) [@lin2014microsoft] settings, the metric combination `mAP@[.55 | medium | 10]` is not included in the default `compute` output but can be obtained using the `mean_evaluator`, after calling `compute`.

```py title="mean_evaluator_example"
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
output = metrics.compute(y_true, y_pred, extended_summary=True)
mean_evaluator = output["mean_evaluator"]
_metric = mean_evaluator(
    iou_threshold=.55,
    max_detection_threshold=10,
    area_range_key="medium",
    metrics="AP"
    )
print(_metric)
"""
{'mAP@[.55 | medium | 10]': 0.16831683168316827}
"""
```
For a complete list of arguments accepted by the `mean_evaluator` function, refer to the `extended_summary` option in the [ODMetrics.compute()][src.od_metrics.od_metrics.ODMetrics.compute] method.

## `IoU`
The calculation of [mAP](map_mar.md#what-is-map) and [mAR](map_mar.md#average-recall) relies on [IoU](iou.md) (Intersection over Union). You can use the standalone `iou` function from `OD-Metrics`.
```py title="iou_example"
from od_metrics import iou

y_true = [[25, 16, 38, 56], [129, 123, 41, 62]]
y_pred = [[25, 27, 37, 54], [119, 111, 40, 67], [124, 9, 49, 67]]

result = iou(y_true, y_pred, box_format="xywh")
print(result)
"""
array([[0.67655425, 0.        ],
       [0.        , 0.46192609],
       [0.        , 0.        ]])
"""
```
The `iou` function supports the `iscrowd` parameter from the [COCOAPI](https://github.com/cocodataset/cocoapi). For more details, refer to the [iscrowd](iou.md#iscrowd-parameter) section.

## References