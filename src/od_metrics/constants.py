"""Constants module."""

from __future__ import annotations

__all__ = [
    "DEFAULT_COCO",
    "_STANDARD_OUTPUT",
    ]

import numpy as np

# Default COCO parameters
DEFAULT_COCO: dict[str, np.ndarray | dict[str, list[float]]] = {
    "iou_thresholds": np.linspace(
        start=.5,
        stop=0.95,
        num=int(np.round((0.95 - .5) / .05)) + 1,
        endpoint=True
        ),
    "recall_thresholds": np.linspace(
        start=.0,
        stop=1.00,
        num=int(np.round((1.00 - .0) / .01)) + 1,
        endpoint=True
        ),
    "max_detection_thresholds": np.array([1, 10, 100]),
    "area_ranges": {
        "all": [0 ** 2, 1e5 ** 2],
        "small": [0 ** 2, 32 ** 2],
        "medium": [32 ** 2, 96 ** 2],
        "large": [96 ** 2, 1e5 ** 2]
        },
    "class_metrics": True,
    }


# Standard metrics output
_STANDARD_OUTPUT: list[dict[str, str | int | float | None]] = [
    {
     "iou_threshold": None,
     "area_range_key": "all",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": .5,
     "area_range_key": "all",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": .75,
     "area_range_key": "all",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": None,
     "area_range_key": "small",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": None,
     "area_range_key": "small",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": None,
     "area_range_key": "medium",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": None,
     "area_range_key": "large",
     "max_detection_threshold": 100,
     "metrics": None,
     },
    {
     "iou_threshold": None,
     "area_range_key": "all",
     "max_detection_threshold": 1,
     "metrics": "AR",
     },
    {
     "iou_threshold": None,
     "area_range_key": "all",
     "max_detection_threshold": 10,
     "metrics": "AR",
     },
    {
     "iou_threshold": None,
     "area_range_key": "all",
     "max_detection_threshold": 100,
     "metrics": "AR",
     },
    ]
