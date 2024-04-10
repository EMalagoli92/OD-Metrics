"""Utility module."""

from __future__ import annotations

__all__ = [
    "_Missing",
    "to_array",
    "get_indexes",
    "get_suffix",
    ]

from typing import Literal, Any
import numpy as np


class _Missing:
    """Sentinel class for missing values."""


def to_array(
        input_: Any,
        ) -> np.ndarray:
    """
    Trasform input to `np.ndarray`.

    Parameters
    ----------
    input_ : Any | None, optional
        Input to be converted.

    Returns
    -------
    np.ndarray
        Input converted to `np.ndarray`.
    """
    if not isinstance(input_, np.ndarray):
        output = np.array(input_)
    else:
        output = input_

    if output.ndim == 0:
        output = output.reshape(-1)
    return output


def get_indexes(
        array1: np.ndarray,
        array2: np.ndarray
        ) -> np.ndarray:
    """
    Get a list of indices.

    Returns a list of indices where elements from `array1` are present
    in `array2`.

    Parameters
    ----------
    array1 : np.ndarray
        The `np.ndarray` to search for elements.
    array2 : np.ndarray
        The `np.ndarray` to search for matches.

    Returns
    -------
    np.ndarray
        A `np.ndarray` of indices where elements from `array1` are
        present in `array2`.
    """
    return np.sort(np.where(np.isin(
        array1, array2))[0])


def xyxy_xywh(bbox: list[float]) -> list[float]:
    """
    Change bounding box format from `xyxy` to `xywh`.

    Parameters
    ----------
    bbox : list[float]
        Input bounding box.

    Returns
    -------
    list[float]
        Bounding box in `"xywh"` format.
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
        ]


def cxcywh_xywh(bbox: list[float]) -> list[float]:
    """
    Change bounding box format from `"cxcywh"` to `"xywh"`.

    Parameters
    ----------
    bbox : list[float]
        Input bounding box.

    Returns
    -------
    list[float]
        Bounding box in `"xywh"` format.
    """
    return [
        bbox[0] - bbox[2] / 2,
        bbox[1] - bbox[3] / 2,
        bbox[2],
        bbox[3]
        ]


def to_xywh(
        bbox: list[float],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[float]:
    """
    Change bounding box format to `"xywh"`.

    Parameters
    ----------
    bbox : list[float]
        Input bounding box.
    box_format : Literal["xyxy", "xywh", "cxcywh"]
        Input bounding box format.
        It can be `"xyxy"`, `"xywh"` or `"cxcywh"`.

    Raises
    ------
    ValueError
        If `box_format` not one of `"xyxy"`, `"xywh"`, `"cxcywh"`.

    Returns
    -------
    list[float]
        Bounding box in `"xywh"` format.
    """
    if box_format == "xywh":
        return bbox
    if box_format == "xyxy":
        return xyxy_xywh(bbox)
    if box_format == "cxcywh":
        return cxcywh_xywh(bbox)
    raise ValueError(  # pragma: no cover
        "`box_format` can be `'xyxy'`, `'xywh'`, `'cxcywh'`. "
        f"Found {box_format}"
        )


def get_suffix(
        iou_threshold: np.ndarray,
        area_range_key: np.ndarray,
        max_detection_threshold: np.ndarray,
        ) -> str:
    """
    Create metric's suffix.

    The format is:
        @[iou_threshold | area_range_key | max_detection_threshold]

    Parameters
    ----------
    iou_threshold : np.ndarray
        IoU threshold.
    area_range_key : np.ndarray
        Area range.
    max_detection_threshold : np.ndarray
        Threshold on maximum detections per image.

    Returns
    -------
    str
        Metric's suffix.
    """
    str_name = []
    # Iou threshold
    lower_bound = str(iou_threshold[0]).replace("0.", ".")
    if iou_threshold.shape[0] > 1:
        upper_bound = str(iou_threshold[-1]).replace("0.", ".")
        str_name += [f"{lower_bound}:{upper_bound}"]
    else:
        str_name += [f"{lower_bound}"]

    # Area range label
    if not np.equal(area_range_key, np.array([None])).all():
        str_name += ["_".join(area_range_key.tolist())]

    # Max detection threshold
    if not np.equal(max_detection_threshold, np.array([None])).all():
        str_name += ["_".join(map(str, max_detection_threshold.tolist()))]

    return f"@[{' | '.join(str_name)}]"
