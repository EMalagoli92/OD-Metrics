"""Utility functions module."""

from __future__ import annotations

__all__ = [
    "annotations_generator",
    "pycoco_converter",
    "test_equality",
    "rename_dict",
    "xywh_to",
    "apply_function",
    ]

from typing import Literal, Callable
import random
import numpy as np

np.random.seed(123)


def annotations_generator(
        n_images: int = 10,
        n_classes: int = 3,
        min_objects: int = 0,
        max_objects: int = 10,
        xy_min: int = 0,
        xy_max: int = 100,
        min_box_width: int = 0,
        max_box_width: int = 100,
        min_box_height: int = 0,
        max_box_height: int = 100,
        include_score: bool = False,
        iscrowd_percentage: float = .1,
        ignore_percentage: float = .1,
        ) -> list[dict[Literal["boxes", "labels", "scores",
                               "iscrowd"], np.ndarray]]:
    """
    Annotations generator of Od-metrics format.

    Bounding box are in `"xywh"` format.

    Parameters
    ----------
    n_images : int, optional
        Number of images.
        The default is `10`.
    n_classes : int, optional
        Number of classes.
        The default is `3`.
    min_objects : int, optional
        Minimum number of objects in a single image.
        The default is `0`.
    max_objects : int, optional
        Max number of objects in a single image.
        The default is `10`.
    xy_min : int, optional
        Minimum value for x, y coordinate in bounding box format `"xywh"`.
        The default is `0`.
    xy_max : int, optional
        Minimum value for x, y coordinate in bounding box format `"xywh"`.
        The default is `100`.
    min_box_width : int, optional
        Minimum value for width coordinate in bounding box format `"xywh"`.
        The default is `0`.
    max_box_width : int, optional
        Maximum value for width coordinate in bounding box format `"xywh"`.
        The default is `100`.
    min_box_height : int, optional
        Minimum value for height coordinate in bounding box format `"xywh"`.
        The default is `0`.
    max_box_height : int, optional
        Maximum value for width coordinate in bounding box format `"xywh"`.
        The default is `100`.
    include_score : bool, optional
        Whether to include score.
        The default is `False`.
    iscrowd_percentage: float, optional
        Percentage of images that contains at least one `iscrowd` region.
        The default is `.1`.

    Returns
    -------
    list[dict[Literal["boxes", "labels", "scores", "iscrowd"], np.ndarray]]
        List of dictionaries.
    """
    annotations = []
    iscrowd_indexes = random.sample(list(range(n_images)),
                                    int(iscrowd_percentage * n_images))
    ignore_indexes = random.sample(list(range(n_images)),
                                   int(ignore_percentage * n_images))
    for index in range(n_images):
        tmp = {}
        n_objects = np.random.randint(min_objects, max_objects + 1)
        tmp["labels"] = np.random.randint(0, n_classes, n_objects)
        xy_coord = np.random.randint(xy_min, xy_max, (n_objects, 2))
        width = np.random.randint(min_box_width, max_box_width + 1,
                                  (n_objects, 1))
        height = np.random.randint(min_box_height, max_box_height + 1,
                                   (n_objects, 1))
        tmp["boxes"] = np.hstack([xy_coord, width, height])
        if include_score:
            tmp["scores"] = np.random.random(n_objects)
        if (index in iscrowd_indexes) and (n_objects > 0):
            # iscrowd
            n_iscrowd = np.random.randint(1, n_objects + 1)
            tmp["iscrowd"] = np.hstack([np.ones(n_iscrowd),
                                        np.zeros(n_objects - n_iscrowd)])
        if (index in ignore_indexes) and (n_objects > 0):
            # ignore
            n_ignore = np.random.randint(1, n_objects + 1)
            tmp["ignore"] = np.hstack([np.ones(n_ignore),
                                       np.zeros(n_objects - n_ignore)])
        annotations.append(tmp)
    return annotations


def pycoco_converter(
        annotations: list[dict[Literal[
            "boxes", "labels", "scores"], np.ndarray]]
        ) -> dict[Literal["images", "annotations", "categories"], list]:
    """
    Convert od-metrics annotations to pycoco format.

    Parameters
    ----------
    annotations : list[dict[Literal["boxes", "labels", "scores"], np.ndarray]]
        Annotations in od-metrics format.

    Returns
    -------
    dict[Literal["images", "annotations", "categories"], list]
        pycoco annotations.
    """
    pycoco_annotations = []
    annotation_id = 1
    for image_id, image_annotation in enumerate(annotations):
        n_image_annotation = len(image_annotation["labels"])
        for i in range(n_image_annotation):
            _annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "area": (image_annotation["boxes"][i][2]
                         * image_annotation["boxes"][i][3]),
                "category_id": image_annotation["labels"][i],
                "iscrowd": (
                    image_annotation["iscrowd"][i]
                    if "iscrowd" in image_annotation.keys() else 0
                    ),
                "ignore": (
                    image_annotation["ignore"][i]
                    if "ignore" in image_annotation.keys() else 0
                    ),
                "bbox": image_annotation["boxes"][i],
                }
            scores = image_annotation.get("scores", None)
            if scores is not None:
                _annotation["score"] = scores[i]

            pycoco_annotations.append(_annotation)

            annotation_id += 1

    _labels = np.unique(np.concatenate([
        _["labels"] for _ in annotations]).flatten()).tolist()

    output = {
        "images": [{"id": id_} for id_ in range(len(annotations))],
        "annotations": pycoco_annotations,
        "categories": [{"id": id_} for id_ in _labels],
        }

    return output


def test_equality(
        input1: np.int64 | np.ndarray | dict | list,
        input2: np.int64 | np.ndarray | dict | list,
        ) -> bool:
    """
    Recursively test equality between two inputs.

    Parameters
    ----------
    input1 : np.int64 | np.ndarray | dict | list
        First input.
    input2 : np.int64 | np.ndarray | dict | list
        Second input.

    Raises
    ------
    ValueError
        If `input1` and `input2` have different types.
    AssertionError
        If `input1` and `input2` have different length.

    Returns
    -------
    bool
        Whether the two inputs are equal.
    """
    try:
        if isinstance(input1, np.int64):
            input1 = int(input1)
        if isinstance(input2, np.int64):
            input2 = int(input2)

        if not isinstance(input1, type(input2)):
            raise ValueError(f"Found: {type(input1)} and {type(input2)}")

        checks = []
        if isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):
            checks.append(np.array_equal(input1, input2))

        elif isinstance(input1, dict) and isinstance(input2, dict):
            assert len(input1) == len(input2), ("Two inputs have different "
                                                "length.")
            for key in input1.keys():
                checks.append(test_equality(input1[key], input2[key]))
        elif isinstance(input1, list) and isinstance(input2, list):
            assert len(input1) == len(input2), ("Two inputs have different "
                                                "length.")

            for el1, el2 in zip(input1, input2):
                checks.append(test_equality(el1, el2))
        else:
            checks.append(input1 == input2)
    except AssertionError:
        checks = [False]

    return all(checks)


def rename_dict(
        dict_: dict,
        rename_dict_keys: dict
        ) -> dict:
    """
    Rename dictionary keys.

    Parameters
    ----------
    dict_ : dict
        Dictionary whose keys have to be renamed.
    rename_dict_keys : dict
        Keys rename dictionary.

    Returns
    -------
    dict
        Renamed dictionary.
    """
    for key, value in rename_dict_keys.items():
        dict_[value] = dict_[key]
        del dict_[key]

    return dict_


def xywh_xyxy(bbox: list[float]) -> list[float]:
    """
    Change bounding box format from `xywh` to `xyxy`.

    Parameters
    ----------
    bbox : list[float]
        Input bounding box.

    Returns
    -------
    list[float]
        Bounding box in `"xyxy"` format.
    """
    return [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
        ]


def xywh_cxcywh(bbox: list[float]) -> list[float]:
    """
    Change bounding box format from `"xywh"` to `"cxcywh"`.

    Parameters
    ----------
    bbox : list[float]
        Input bounding box.

    Returns
    -------
    list[float]
        Bounding box in `"cxcywh"` format.
    """
    return [
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] / 2,
        bbox[2],
        bbox[3]
        ]


def xywh_to(
        bbox: list[float],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[float]:
    """
    Change bounding box format from `"xywh"` to `box_format`.

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
        If `box_format` not in `"xyxy"`, `"xywh"`, `"cxcywh"`.

    Returns
    -------
    list[float]
        Bounding box in `box_format` format.
    """
    if box_format == "xywh":
        return bbox
    if box_format == "xyxy":
        return xywh_xyxy(bbox)
    if box_format == "cxcywh":
        return xywh_cxcywh(bbox)
    raise ValueError("`box_format` can be `'xyxy'`, `'xywh'`, `'cxcywh'`. "
                     f"Found {box_format}")


def apply_function(
        x: list | np.ndarray,
        func: Callable,
        ) -> list | np.ndarray:
    """
    Apply a function to every element of a `list` or `np.ndarray`.

    Parameters
    ----------
    x : list | np.ndarray
        Input data. It can be a `list` or `np.ndarray`.
    func : Callable
        Function to apply to every element of input `x`.

    Raises
    ------
    TypeError
        If input is neither a `list` or `np.ndarray`.

    Returns
    -------
    list | np.ndarray
        Input with function applied to every element of `x`.
    """
    if isinstance(x, list):
        return [func(elem) for elem in x]
    if isinstance(x, np.ndarray):
        return np.array([func(elem) for elem in x])
    raise TypeError("Type not supported. Supported types are: `list` or"
                    f"`np.ndarray`. Found: {type(x)}")
