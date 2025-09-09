"""Validators module."""

from __future__ import annotations

__all__ = [
    "ConstructorModel",
    "ComputeModel",
    "MeanModel",
    ]

from functools import reduce
from collections import Counter
from typing import Any, Literal, Optional
from typing_extensions import Self
import numpy as np

from pydantic import BaseModel, field_validator, ValidationInfo, ConfigDict

from .utils import to_xywh


def _common_validator(
        name: str,
        value: Any,
        default_flag: Any,
        default_value: np.ndarray,
        dtype: type[int] | type[str] | type[float],
        allow_none: bool,
        ) -> np.ndarray:
    """
    Compute common validation.

    Parameters
    ----------
    name : str
        Field name
    value : Any
        Input value.
    default_flag : Any
        Flag for default value.
    default_value : np.ndarray
        Default value.
    dtype : type[int] | type[str] | type[float]
        Value dtypes.
    allow_none : bool
        Whether to allow `None` values.

    Raises
    ------
    ValueError
        If `value` is neither a `dtype`, a list of `dtype` or
        a 1-dimensional `np.ndarray`, or `None` if `allow_none=True`.
        If `value` contains duplicate values.

    Returns
    -------
    np.ndarray
    """
    # dtype
    if isinstance(value, dtype):
        _value = np.array([value])

    # list[dtype]
    elif (
            isinstance(value, list)
            and all(isinstance(x, dtype) for x in value)
            ):
        _value = np.array(value)

    # np.ndarray
    elif isinstance(value, np.ndarray) and (value.ndim == 1):
        _value = value

    # default
    elif value is default_flag:
        _value = default_value

    # None
    elif (value is None) and allow_none:
        _value = np.array([None])

    else:
        _msg = (
            f"Invalid value for {name}. "
            f"{name} should be a {dtype.__name__}, list of {dtype.__name__} "
            "or a 1-dimensional `np.ndarray`."
            )
        _msg += "or `None`." if allow_none else "."
        raise ValueError(_msg)

    # Check duplicates
    has_duplicates = any(count > 1 for count in
                         np.unique(_value, return_counts=True)[1])

    if has_duplicates:
        raise ValueError(
            f"Duplicate values found in {name}. "
            f"{name} should not contain any duplicate values."
            )

    return _value


def _area_ranges_validator(
        name: str,
        value: Any,
        default_flag: Any,
        default_value: dict[str, list[float]]
        ) -> dict[str | None, list[float]]:
    """
    Validate `area_ranges` field.

    Parameters
    ----------
    name : str
        Field name.
    value : Any
        Input value.
    default_flag : Any
        Flag for default value.
    default_value :  dict[str, list[float]]
        Default value.

    Raises
    ------
    ValueError
        If `value` is neither a dictionary with string keys and values
        as lists/`np.ndarray` of 2 integers/floats, with first element less or
        equal than second, or `None`.
        If `value` contains duplicate values.

    Returns
    -------
    dict[str | None, list[float]]
    """
    # dict[,np.ndarray] --> dict[, list]
    if (
            isinstance(value, dict)
            and (all(isinstance(v, np.ndarray) for v in value.values()))
            ):
        value = {key: value.tolist() for key, value in value.items()}
    # dict[str, list[float]]
    if (
            isinstance(value, dict)  # pylint: disable=R0916
            and (all(isinstance(k, str) for k in value.keys()))
            and (all(isinstance(v, list) for v in value.values()))
            and (all(map(lambda x: len(x) == 2, value.values())))
            and all(isinstance(x, (int, float)) for x in
                    reduce(lambda x, y: x + y, value.values()))
            and all(map(lambda x: x[0] <= x[1], value.values()))
            ):
        _value = value
    # None
    elif value is None:
        _value = {None: [float("-inf"), float("inf")]}
    # Missing
    elif value is default_flag:
        _value = default_value
    else:
        raise ValueError(
            f"Invalid value for {name}. "
            f"{name} should be either:\n"
            "- A dictionary with string keys and values as lists/`np.ndarray` "
            "of 2 integers/floats, with first element less or equal than "
            "second.\n"
            "- `None`.\n"
            )
    # Check duplicates in .values()
    _values = [tuple(x) if isinstance(x, list) else tuple([x])
               for x in _value.values()]
    has_duplicates = any(
        count > 1 for count in Counter(_values).values())

    if has_duplicates:
        raise ValueError(
            f"Duplicate values found in {name}. "
            f"{name} should not contain any duplicate values."
            )
    return _value


def _class_metrics_validator(
        name: str,
        value: Any,
        default_flag: Any,
        default_value: bool
        ) -> bool:
    """
    Validate `class_metrics` field.

    Parameters
    ----------
    name : str
        Field name.
    value : Any
        Input value.
    default_flag : Any
        Flag for default value.
    default_value :  bool
        Default value.

    Raises
    ------
    ValueError
        If `value` is not a `bool`.

    Returns
    -------
    bool
    """
    # bool
    if isinstance(value, bool):
        _value = value
    # Missing
    elif value is default_flag:
        _value = default_value
    else:
        raise ValueError(
            f"Invalid value for {name}. {name} should be a `bool`.")

    return _value


class ConstructorModel(BaseModel):
    """`__init__()` method Model."""

    iou_thresholds: np.ndarray
    recall_thresholds: np.ndarray
    max_detection_thresholds: np.ndarray
    area_ranges: dict[Optional[str], list[float]]
    class_metrics: bool
    box_format: Literal["xyxy", "xywh", "cxcywh"]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        strict=True
        )

    @field_validator("iou_thresholds", "recall_thresholds", mode="before")
    @classmethod
    def iou_recall_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo
            ) -> np.ndarray:
        """
        Validate `iou_thresholds` and `recall_thresholds` fields.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Raises
        ------
        ValueError
            If context or field name informations are missing.

        Returns
        -------
        np.ndarray
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _common_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name],
            dtype=float,
            allow_none=False,
            )

    @field_validator("max_detection_thresholds", mode="before")
    @classmethod
    def max_detection_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo
            ) -> np.ndarray:
        """
        Validate `max_detection_thresholds` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Raises
        ------
        ValueError
            If context or field name informations are missing.

        Returns
        -------
        np.ndarray
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _common_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name],
            dtype=int,
            allow_none=True,
            )

    @field_validator("area_ranges", mode="before")
    @classmethod
    def area_ranges_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo,
            ) -> dict[str | None, list[float]]:
        """
        Validate `area_ranges` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Raises
        ------
        ValueError
            If context or field name informations are missing.

        Returns
        -------
        dict[str | None, list[float]]
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _area_ranges_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name]
            )

    @field_validator("class_metrics", mode="before")
    @classmethod
    def class_metrics_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo,
            ) -> bool:
        """
        Validate `class_metrics` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Raises
        ------
        ValueError
            If context or field name informations are missing.

        Returns
        -------
        bool
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _class_metrics_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name]
            )


def _reformat_y_true(
        data: list[dict],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[dict]:
    """
    Reformat ground truth annotations.

    Parameters
    ----------
    data : list[dict]
        List of per-image annotations.
    box_format : Literal["xyxy", "xywh", "cxcywh"]
        Format of the input bounding boxes.

    Returns
    -------
    list[dict]
        Reformatted list of annotations.
    """
    annotations = []
    id_ = 1
    for image_id, image_anns in enumerate(data):
        n_annotations = len(image_anns["boxes"])
        for i in range(n_annotations):
            # Common
            annotation_tmp: dict = {
                "id": id_,
                "image_id": image_id,
                "label_id": image_anns["labels"][i],
                "bbox": to_xywh(
                    bbox=image_anns["boxes"][i],
                    box_format=box_format,
                    )
                }
            annotation_tmp["area"] = (
                image_anns["area"][i] if image_anns["area"][i] is not None
                else (annotation_tmp["bbox"][2]
                      * annotation_tmp["bbox"][3])
                )

            annotation_tmp["iscrowd"] = image_anns["iscrowd"][i]
            annotation_tmp["ignore"] = image_anns["iscrowd"][i]

            id_ += 1
            annotations.append(annotation_tmp)

    return annotations


def _reformat_y_pred(
        data: list[dict],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[dict]:
    """
    Reformat predicted annotations.

    Parameters
    ----------
    data : list[dict]
        List of per-image predictions.
    box_format : Literal["xyxy", "xywh", "cxcywh"]
        Format of the input bounding boxes.

    Returns
    -------
    list[dict]
        Reformatted list of predictions.
    """
    annotations = []
    id_ = 1
    for image_id, image_anns in enumerate(data):
        n_annotations = len(image_anns["boxes"])
        for i in range(n_annotations):
            # Common
            annotation_tmp: dict = {
                "id": id_,
                "image_id": image_id,
                "label_id": image_anns["labels"][i],
                "bbox": to_xywh(
                    bbox=image_anns["boxes"][i],
                    box_format=box_format,
                    )
                }
            annotation_tmp["area"] = (
                image_anns["area"][i] if image_anns["area"][i] is not None
                else (annotation_tmp["bbox"][2]
                      * annotation_tmp["bbox"][3])
                )
            annotation_tmp |= {"score": image_anns["scores"][i]}
            id_ += 1
            annotations.append(annotation_tmp)

    return annotations


def _preprocess(
        data: dict,
        mode: Literal["y_true", "y_pred"],
        ) -> dict:
    """
    Preprocess a single annotation dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing a single annotation.
    mode : Literal["y_true", "y_pred"]
        Type of annotation to preprocess.

    Returns
    -------
    dict
        Preprocessed annotation dictionary.
    """
    # Check base keys: `boxes`, `labels`
    _base_keys = ["boxes", "labels"]
    diff = set(_base_keys) - set(data.keys())
    if diff:
        msg = f"Missing base keys in annotation: {diff}"
        raise ValueError(msg)

    # Preprocess `boxes`
    if not all(map(lambda x: len(x) == 4, data["boxes"])):
        raise ValueError("Each box must have 4 elements.")
    if isinstance(data["boxes"], np.ndarray):
        data["boxes"] = data["boxes"].tolist()

    # Preprocess `labels`
    if isinstance(data["labels"], np.ndarray):
        data["labels"] = data["labels"].tolist()

    # Number of objects
    length = len(data["boxes"])

    if mode == "y_true":
        # Add `iscrowd` and `ignore` fields
        for field in ["iscrowd", "ignore"]:
            if field not in data.keys():
                data[field] = [False] * length
            elif isinstance(data[field], np.ndarray):
                data[field] = data[field].tolist()
            _base_keys += [field]

        # Preprocess `iscrowd` and `ignore`
        for field in ["iscrowd", "ignore"]:
            data[field] = list(map(bool, data[field]))

    elif mode == "y_pred":
        # Preprocess `scores`
        if isinstance(data["scores"], np.ndarray):
            data["scores"] = data["scores"].tolist()

    # Add `area` field
    if "area" not in data.keys():
        data["area"] = [None] * length
    _base_keys += ["area"]
    # Preprocess `area`
    if isinstance(data["area"], np.ndarray):
        data["area"] = data["area"].tolist()

    # Check fields length
    lengths = [len(data[field]) for field in _base_keys]
    if len(set(lengths)) != 1:
        msg = (f"All fields {_base_keys} in a single annotation "
               "must have the same length, matching the number of boxes.")
        raise ValueError(msg)

    return data


class ComputeModel:
    """`compute` method Model."""

    @staticmethod
    def model_validate(
        y_true: list[dict],
        y_pred: list[dict],
        extended_summary: bool,
        box_format: Literal["xyxy", "xywh", "cxcywh"]
        ) -> tuple[list[dict], list[dict]]:
        """
        Validate and preprocess ground truth and predictions.

        Parameters
        ----------
        y_true : list[dict]
            Ground truth annotations.
        y_pred : list[dict]
            Predicted annotations.
        extended_summary : bool
            Whether to return an extended summary.
        box_format : Literal["xyxy", "xywh", "cxcywh"]
            Format of the input bounding boxes.

        Returns
        -------
        tuple[list[dict], list[dict]]
            Preprocessed ground truth and predictions.
        """
        if not isinstance(extended_summary, bool):
            msg = "`extended_summary` should be `bool`."
            raise TypeError(msg)

        # Check lengths
        if len(y_true) != len(y_pred):
            msg = "`y_true` and `y_pred` must have the same length."
            raise ValueError(msg)

        # y_true
        y_true = [_preprocess(data=data, mode="y_true") for data in y_true]
        y_true = _reformat_y_true(
            data=y_true,
            box_format=box_format,
            )

        # y_pred
        y_pred = [_preprocess(data=data, mode="y_pred") for data in y_pred]
        y_pred = _reformat_y_pred(
            data=y_pred,
            box_format=box_format,
        )

        return y_true, y_pred


class MeanModel(BaseModel):
    """Mean evaluator Model."""

    iou_threshold: np.ndarray
    area_range_key: np.ndarray
    max_detection_threshold: np.ndarray
    label_id: np.ndarray
    metrics: list[Literal["AP", "AR"]]
    include_spec: bool
    prefix: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        strict=True
        )

    @field_validator("iou_threshold", mode="before")
    @classmethod
    def iou_threshold_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo
            ) -> np.ndarray:
        """
        Validate `iou_threshold` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Returns
        -------
        np.ndarray
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _common_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name],
            dtype=float,
            allow_none=False,
            )

    @field_validator("area_range_key", mode="before")
    @classmethod
    def area_range_key_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo
            ) -> np.ndarray:
        """
        Validate `area_range_key` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Returns
        -------
        np.ndarray
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _common_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name],
            dtype=str,
            allow_none=False,
            )

    @field_validator("max_detection_threshold", "label_id", mode="before")
    @classmethod
    def max_detection_label_id_validator(
            cls: type[Self],
            value: Any,
            info: ValidationInfo
            ) -> np.ndarray:
        """
        Validate `max_detection_threshold` field.

        Parameters
        ----------
        value : Any
            Input value.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Returns
        -------
        np.ndarray
        """
        if (
                info.context is None
                or "default_flag" not in info.context
                or "default_value" not in info.context
                or info.field_name is None
                ):
            raise ValueError(  # pragma: no cover
                "Missing required context or field name information.")

        return _common_validator(
            name=info.field_name,
            value=value,
            default_flag=info.context["default_flag"],
            default_value=info.context["default_value"][info.field_name],
            dtype=int,
            allow_none=False,
            )

    @field_validator("metrics", mode="before")
    @classmethod
    def metrics_validator(
            cls: type[Self],
            value: Any,
            ) -> list[str]:
        """
        Validate `metrics` field.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        list
        """
        if isinstance(value, str):
            value_ = [value]
        elif value is None:
            value_ = ["AP", "AR"]
        elif isinstance(value, list):
            value_ = value

        return value_
