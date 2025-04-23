"""Validators module."""

from __future__ import annotations

__all__ = [
    "ConstructorModel",
    "ComputeModel",
    "MeanModel",
    ]

from functools import reduce
from collections import Counter
from typing import Any, Literal, Optional, Union
from typing_extensions import Self
import numpy as np

from pydantic import BaseModel, field_validator, model_validator, \
    ValidationInfo, ConfigDict

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
        default_value: dict[str, list[float]]
        ) -> dict[str | None, list[float]]:
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
    default_value :  dict[str, list[float]]
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


class BaseInputAnnotationModel(BaseModel):
    """Base Annotation Model."""

    boxes: list[list[float]]
    labels: list[int]
    area: list[Optional[float]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        strict=True
        )

    @model_validator(mode="after")
    def check_lenghts(self) -> Self:
        """
        Check fields length.

        Raises
        ------
        ValueError
            If model fields don't have the same length.

        Returns
        -------
        Self
        """
        lenghts = [len(getattr(self, field)) for field in self.model_fields]
        if len(set(lenghts)) != 1:
            raise ValueError(
                f"All {self.model_fields} in a single annotation "
                "must have the same length, i.e. the number "
                "of boxes."
                )
        return self

    @field_validator("boxes", mode="before")
    @classmethod
    def validate_boxes(
            cls: type[Self],
            value: Union[list[list[float]], np.ndarray],
            ) -> list:
        """
        Validate `boxes` field.

        Check that each box value has 4 elements and convert it to list.

        Parameters
        ----------
        value : Union[list[list[float]], np.ndarray]
            Boxes.

        Raises
        ------
        ValueError
            If one box does not have 4 elements.

        Returns
        -------
        list
        """
        if not all(map(lambda x: len(x) == 4, value)):
            raise ValueError(
                "Each box must have 4 elements."
                )
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(
            cls: type[Self],
            value: Union[list, np.ndarray],
            ) -> list:
        """
        Validate `labels` field.

        Parameters
        ----------
        value : Union[list, np.ndarray]
            Input value.

        Returns
        -------
        list
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_area(
            cls: type[Self],
            data: dict,
            ) -> dict:
        """
        Validate `area` field.

        Parameters
        ----------
        data : dict
            Data values.

        Raises
        ------
        ValueError
            If `boxes` key not in `data`.

        Returns
        -------
        dict
            Data with `area` field.
        """
        if "boxes" not in data:
            raise ValueError("`boxes` must be in data.")
        length = len(data["boxes"])
        if "area" not in data.keys():
            data["area"] = [None] * length
        elif isinstance(data["area"], np.ndarray):
            data["area"] = data["area"].tolist()
        return data


class YTrueInputModel(BaseInputAnnotationModel):
    """Ground truth input annotations Model."""

    iscrowd: list[bool]
    ignore: list[bool]

    @model_validator(mode="before")
    @classmethod
    def validate_iscrowd_ignore(
            cls: type[Self],
            data: dict,
            ) -> dict:
        """
        Validate `iscrowd` and `ignore` fields.

        Parameters
        ----------
        data : dict
            Data values.

        Raises
        ------
        ValueError
            If `boxes` not in `data`.

        Returns
        -------
        dict
            Data with `iscrowd` and `ignore` fields.
        """
        if "boxes" not in data:
            raise ValueError("`boxes` must be in data.")
        length = len(data["boxes"])

        for field in ["iscrowd", "ignore"]:
            if field not in data.keys():
                data[field] = [False] * length
            elif isinstance(data[field], np.ndarray):
                data[field] = data[field].tolist()

            data[field] = list(map(bool, data[field]))

        return data


class YPredInputModel(BaseInputAnnotationModel):
    """Prediction input annotations Model."""

    scores: list[float]

    @field_validator("scores", mode="before")
    @classmethod
    def validate_scores(
            cls: type[Self],
            value: Union[list, np.ndarray],
            ) -> list:
        """
        Validate `scores` field.

        Parameters
        ----------
        value : Union[list, np.ndarray]
            Input value.

        Returns
        -------
        list
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value


class YTrueOutputModel(BaseModel):
    """Groundtruth output annotations Model."""

    id: int  # pylint: disable=C0103
    image_id: int
    label_id: int
    bbox: list[float]
    area: float
    ignore: bool
    iscrowd: bool

    model_config = ConfigDict(strict=True)


class YPredOutputModel(BaseModel):
    """Prediction output annotations Model."""

    id: int  # pylint: disable=C0103
    image_id: int
    label_id: int
    bbox: list[float]
    area: float
    score: float

    model_config = ConfigDict(strict=True)


def _convert_y_true(
        data: list[YTrueInputModel],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[YTrueOutputModel]:
    """
    Convert a list of `YTrueInputModel` to a list of `YTrueOutputModel`.

    Parameters
    ----------
    data : list[YTrueInputModel]
        List of `YTrueInputModel`.
    box_format : Literal["xyxy", "xywh", "cxcywh"]
        Input format of given boxes.

    Returns
    -------
    list[YTrueOutputModel]
        List of `YTrueOutputModel`.
    """
    annotations = []
    id_ = 1
    for image_id, image_anns in enumerate(data):
        n_annotations = len(image_anns.boxes)
        for i in range(n_annotations):
            # Common
            annotation_tmp: dict = {
                "id": id_,
                "image_id": image_id,
                "label_id": image_anns.labels[i],
                "bbox": to_xywh(
                    bbox=image_anns.boxes[i],
                    box_format=box_format,
                    )
                }
            annotation_tmp["area"] = (
                image_anns.area[i] if image_anns.area[i] is not None
                else (annotation_tmp["bbox"][2]
                      * annotation_tmp["bbox"][3])
                )

            annotation_tmp["iscrowd"] = image_anns.iscrowd[i]
            annotation_tmp["ignore"] = image_anns.iscrowd[i]

            id_ += 1
            annotations.append(YTrueOutputModel(**annotation_tmp))

    return annotations


def _convert_y_pred(
        data: list[YPredInputModel],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        ) -> list[YPredOutputModel]:
    """
    Convert a list of `YPredInputModel` to a list of `YPredOutputModel`.

    Parameters
    ----------
    data : list[YPredInputModel]
        List of `YPredInputModel`.
    box_format : Literal["xyxy", "xywh", "cxcywh"]
        Input format of given boxes.

    Returns
    -------
    list[YPredOutputModel]
        List of `YPredOutputModel`.
    """
    annotations = []
    id_ = 1
    for image_id, image_anns in enumerate(data):
        n_annotations = len(image_anns.boxes)
        for i in range(n_annotations):
            # Common
            annotation_tmp: dict = {
                "id": id_,
                "image_id": image_id,
                "label_id": image_anns.labels[i],
                "bbox": to_xywh(
                    bbox=image_anns.boxes[i],
                    box_format=box_format,
                    )
                }
            annotation_tmp["area"] = (
                image_anns.area[i] if image_anns.area[i] is not None
                else (annotation_tmp["bbox"][2]
                      * annotation_tmp["bbox"][3])
                )
            annotation_tmp |= {"score": image_anns.scores[i]}
            id_ += 1
            annotations.append(YPredOutputModel(**annotation_tmp))

    return annotations


class ComputeModel(BaseModel):
    """`compute` method Model."""

    y_true: list[YTrueOutputModel]
    y_pred: list[YPredOutputModel]
    extended_summary: bool

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        strict=True
        )

    @model_validator(mode="before")
    @classmethod
    def annotation_parser(
            cls: type[Self],
            data: dict,
            info: ValidationInfo,
            ) -> dict:
        """
        Parse annotations.

        Parameters
        ----------
        data : dict
            Input annotations.
        info : ValidationInfo
            Pydantic `ValidationInfo`.

        Returns
        -------
        dict
            Ground truth or predictions annotations.
        """
        if info.context is None or "box_format" not in info.context:
            raise ValueError(  # pragma: no cover
                "Missing required context or `box_format` information.")
        box_format = info.context["box_format"]

        # y_true
        y_true_input = [YTrueInputModel(**v) for v in data["y_true"]]
        y_pred_input = [YPredInputModel(**v) for v in data["y_pred"]]

        if len(y_true_input) != len(y_pred_input):
            raise ValueError(
                    "Expected argument `y_true` and `y_pred` "
                    "to have the same length, i.e. same number "
                    "of images."
                    )

        data["y_true"] = _convert_y_true(
            y_true_input,
            box_format=box_format
            )
        data["y_pred"] = _convert_y_pred(
            y_pred_input,
            box_format=box_format
            )

        return data


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
