"""ODMetrics main class implementation."""  # pylint: disable=C0302

from __future__ import annotations

__all__ = [
    "ODMetrics",
    "iou",
    ]

import copy
from itertools import groupby, product
from functools import reduce, partial
from collections import defaultdict
import operator
from typing import Literal, Any, Callable, cast
import numpy as np

from .constants import DEFAULT_COCO, _STANDARD_OUTPUT
from .utils import to_array, get_indexes, get_suffix, _Missing
from .validators import ConstructorModel, ComputeModel, MeanModel


class ODMetrics:
    """
    ODMetrics class.

    Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)
    for Object Detection.
    """

    def __init__(
            self,
            iou_thresholds: (float | list[float] | np.ndarray |
                             type[_Missing]) = _Missing,
            recall_thresholds: (float | list[float] | np.ndarray |
                                type[_Missing]) = _Missing,
            max_detection_thresholds: (int | list[int] | np.ndarray | None |
                                       type[_Missing]) = _Missing,
            area_ranges: (dict[str, list[float] | np.ndarray] | None |
                          type[_Missing]) = _Missing,
            class_metrics: bool = False,
            box_format: Literal["xyxy", "xywh", "cxcywh"] = "xywh",
            ) -> None:
        """
        Initialize.

        Parameters
        ----------
        iou_thresholds : float | list[float] | np.ndarray \
                          | type[_Missing], optional
            IoU thresholds.
            If not specified (`_Missing`), the default (COCO) is used and
            corresponds to the stepped range `[0.5,...,0.95]` with step
            `0.05` (`10` values).
            The default is `_Missing`.
        recall_thresholds : float | list[float] | np.ndarray \
                             | type[_Missing], optional
            Recall thresholds.
            If not specified (`_Missing`), the default (COCO) is used and
            corresponds to the stepped range `[0,...,1]` with step
            `0.01` (`101` values).
            The default is `_Missing`.
        max_detection_thresholds : int | list[int] | np.ndarray | None \
                                    | type[_Missing], optional
            Thresholds on maximum detections per image.
            If not specified (`_Missing`), the default (COCO) is used and
            corresponds to the list `[1, 10, 100]`.
            If `None`, no limit to detections per image will be set.
            The default is `_Missing`.
        area_ranges : dict[str, list[float] | np.ndarray] \
                       | None | type[_Missing], optional
            Area ranges.
            If not specified, the default (COCO) is used and corresponds to:
            <br>
            `{
                "all": [0 ** 2, 1e5 ** 2],
                "small": [0 ** 2, 32 ** 2],
                "medium": [32 ** 2, 96 ** 2],
                "large": [96 ** 2, 1e5 ** 2]
                }`
            If `None`, no area range limits will be set.
            The default is `_Missing`.
        class_metrics : bool, optional
            Option to enable per-class metrics (See `compute()` method).
            Has a performance impact.
            The default is `False`.
        box_format: Literal["xyxy", "xywh", "cxcywh"], optional
            Bounding box format.
            Supported formats are:<br>
                - `"xyxy"`: boxes are represented via corners,
                        x1, y1 being top left and x2, y2
                        being bottom right.<br>
                - `xywh`: boxes are represented via corner,
                        width and height, x1, y2 being top
                        left, w, h being width and height.
                        This is the default format; all
                        input formats will be converted
                        to this.<br>
                - `"cxcywh"`: boxes are represented via centre,
                        width and height, cx, cy being
                        center of box, w, h being width
                        and height.<br>
            The default is `"xywh"`.

        Returns
        -------
        None
        """
        constructor_model = ConstructorModel.model_validate(
            {
                "iou_thresholds": iou_thresholds,
                "recall_thresholds": recall_thresholds,
                "max_detection_thresholds": max_detection_thresholds,
                "area_ranges": area_ranges,
                "class_metrics": class_metrics,
                "box_format": box_format,
                },
            context={"default_value": DEFAULT_COCO, "default_flag": _Missing}
            )
        self.iou_thresholds: np.ndarray = constructor_model.iou_thresholds
        self.recall_thresholds: np.ndarray = (
            constructor_model.recall_thresholds)
        self.max_detection_thresholds: np.ndarray = (
            constructor_model.max_detection_thresholds)
        self.area_ranges: dict[str | None, list[float]] = (
            constructor_model.area_ranges)
        self.class_metrics: bool = constructor_model.class_metrics
        self.box_format: Literal["xyxy", "xywh", "cxcywh"] = (
            constructor_model.box_format)

    def compute(
            self,
            y_true: list[dict],
            y_pred: list[dict],
            extended_summary: bool = False,
            ) -> dict[str, float | int | list[int]
                      | dict | np.ndarray | partial]:
        """
        Compute metrics.

        Parameters
        ----------
        y_true : list[dict]
            A list consisting of dictionaries each containing
            the key-values: each dictionary corresponds to the ground truth
            of a single image.
            Parameters that should be provided per dict:

                boxes : list[list[float]] | np.ndarray
                    List of floats lists or `np.ndarray`; the length of the
                    list/array correspond to the number of boxes and each
                    list/array is 4-float specifying the box coordinates in
                    the format specified in the constructor.
                labels : list[int] | np.ndarray
                    List of integers or `np.ndarray` specifying the ground
                    truth classes for the boxes: the length corresponds to
                    the number of boxes.
                iscrowd : list[bool | Literal[0, 1]] | np.ndarray
                    List of integers or `np.ndarray` specifying crowd regions:
                    the length corresponds to the number of boxes.
                    The values can be `bool` or `0`/`1` indicating whether the
                    bounding box indicates a crowd of objects.
                    Value is optional, and if not provided it will
                    automatically be set to `False`.
                area : list[float] | np.ndarray
                    A list of `float` or `np.ndarray` specifying the area of
                    the objects: the length corresponds to the number of
                    boxes.
                    Value is optional, and if not provided will
                    be automatically calculated based on
                    the bounding box provided.
                    Only affects which samples contribute the specific
                    area range.
        y_pred : list[dict]
            A list consisting of dictionaries each containing
            the key-values: each dictionary corresponds to the predictions
            of a single image.
            Parameters that should be provided per dict:

                boxes : list[list[float]] | np.ndarray
                    List of float lists or `np.ndarray`; the length of the
                    list/array correspond to the number of boxes and each
                    list/array is 4-float specifying the box coordinates in
                    the format specified in the constructor.
                scores : list[float] | np.ndarray
                    List of floats or `np.ndarray` specifying the score for
                    the boxes: the length corresponds to the number of boxes.
                labels : list[int] | np.ndarray
                    List of integers or `np.ndarray` specifying the ground
                    truth classes for the boxes: the length corresponds to
                    the number of boxes.
        extended_summary : bool, optional
            Option to enable extended summary with additional metrics
            including `IoU`, `AP` (Average Precision), `AR` (Average Recall)
            and `mean_evaluator` (`Callable`).
            The output dictionary will contain the following extra key-values:

                IoU : dict[tuple[int, int], np.ndarray]
                       A dictionary containing the IoU values for every
                       image/class combination e.g. `IoU[(0,0)]`
                       would contain the IoU for image `0` and class `0`.
                       Each value is a `np.ndarray` with shape `(n, m)`
                       where `n` is the number of detections and `m` is
                       the number of ground truth boxes for that image/class
                       combination.
                AP : np.ndarray
                      Average precision: a `np.ndarray` of shape
                      `(T, R, K, A, M)` containing the precision values.
                      Here:
                          - `T` is the number of IoU thresholds
                          - `R` is the number of recall thresholds
                          - `K` is the number of classes
                          - `A` is the number of areas
                          - `M` is the number of max detections per image
                AR : np.ndarray
                     A `np.ndarray` of shape `(T, K, A, M)` containing the
                     averag recall values.
                     Here:
                         - `T` is the number of IoU thresholds
                         - `K` is the number of classes
                         - `A` is the number of areas
                         - `M` is the number of max detections per image
                mean_evaluator : Callable
                    Mean evaluator function.
                    Parameters are:
                        iou_threshold : (float | list[float] | np.ndarray
                                         | None), optional
                            IoU threshold on which calculate the mean.
                            It can be a `float`, a list of floats, `np.ndarray`
                            or `None`; all values must be inlcuded in the
                            constructor argument `iou_thresholds`.
                            If `None`, all input `iou_thresholds` will be used.
                            The default is `None`.
                        area_range_key : (str | list[str] | np.ndarray
                                          | None), optional
                            Area range key on which calculate the mean.
                            It can be a `str`, a list of strings, `np.ndarray`,
                            or `None`; all values must be included in the
                            constructor argument `area_ranges`.
                            If `None`, all input `area_ranges` keys will be
                            used.
                            The default is `None`.
                        max_detection_threshold : (int | list[int] |
                                                   np.ndarray | None), optional
                            Threshold on maxiumum detections per image on
                            which calculate the mean.
                            It can be a `int`, a list of integers, `np.ndarray`
                            or `None`; all values must be inlcuded in the
                            constructor argument `max_detection_thresholds`.
                            If `None`, all input `max_detection_thresholds`
                            will be used.
                            The default is `None`.
                        label_id : (int | list[int] | np.ndarray
                                    | None), optional
                            Label ids on which calculate the mean.
                            If `class_metrics` is `True`, `label_id` must be
                            included in the label ids of the provided `y_true`.
                            If `class_metrics` is `False`, `label_id` must be
                            `-1` (in this case equivalent to `None`).
                            If `None`, all labels will be used.
                            The default is `None`.
                        metrics : (Literal["AP", "AR"]
                                   | list[Literal["AP", "AR"]] | None),
                                   optional
                            Metrics on which calculate the mean.
                            If `None`, both `"AP"` and `"AR"` will be used.
                            The default is `None`.
                        include_spec : bool, optional
                            Whether to include mean settings specification.
                            The default is `False`.
                        prefix : str, optional
                            Prefix to add to metrics keys.
                            The default is `m`.

        Returns
        -------
        dict[str, float | int | list[int] | dict | np.ndarray | partial]
            The format of the output string metric ids is defined as:

            `{metric}@[{iou_thresholds} | {area_ranges}
                      | {max_detection_thresholds}]`

            If a field is `None`, the corresponding string field will be emtpy,
            e.g., `{metric}@[{iou_thresholds} | {area_ranges}]`
            indicate metrics calculated without limit to detections per image,
            i.e. `max_detections_thresholds` set to `None`.
            Assuming that the parameters passed to the constructor
            are the default ones (COCO), the output dictionary will
            contain the following key-values: <br>
                `mAP@[.5 | all | 100]` <br>
                `mAR@[.5 | all | 100]` <br>
                `mAP@[.75 | all | 100]` <br>
                `mAR@[.75 | all | 100]` <br>
                `mAR@[.5:.95 | all | 1]` <br>
                `mAR@[.5:.95 | all | 10]` <br>
                `mAR@[.5:.95 | all | 100]` <br>
                `mAP@[.5:.95 | all | 100]` <br>
                `mAP@[.5:.95 | large | 100]` <br>
                `mAR@[.5:.95 | large | 100]` <br>
                `mAP@[.5:.95 | medium | 100]` <br>
                `mAR@[.5:.95 | medium | 100]` <br>
                `mAP@[.5:.95 | small | 100]` <br>
                `mAR@[.5:.95 | small | 100]` <br>
            If `class_metrics` is `True`, the output dictionary will contain
            the additional key `class_metrics`, a dictionary with class as key
            and value each of the above metrics.
            If `extended_summary` is `True`, the output dictionary will contain
            the additional keys `IoU`, `AP`, `AR` and `mean_evaluator`.
            (See `extended_summary`)
        """
        # Image ids
        images_ids = list(range(len(y_true)))

        # Parse Annotations
        compute_model = ComputeModel.model_validate(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "extended_summary": extended_summary,
                },
            context={"box_format": self.box_format}
            )
        y_true = [y_true_.dict() for y_true_ in compute_model.y_true]
        y_pred = [y_pred_.dict() for y_pred_ in compute_model.y_pred]
        extended_summary = compute_model.extended_summary

        # Get label_ids from y_true
        label_ids = np.unique([_annotation["label_id"]
                               for _annotation in y_true]).tolist()
        _label_ids = label_ids if self.class_metrics else [-1]

        # y_true, y_pred --> default_dict (keys: (image_id, label_id))
        y_true_ddict = defaultdict(
            list,
            [(k, list(v)) for k, v in groupby(
                sorted(y_true, key=lambda x: (x["image_id"], x["label_id"])),
                key=lambda x: (x["image_id"], x["label_id"]))]
            )
        y_pred_ddict = defaultdict(
            list,
            [(k, list(v)) for k, v in groupby(
                sorted(y_pred, key=lambda x: (x["image_id"], x["label_id"])),
                key=lambda x: (x["image_id"], x["label_id"]))]
            )

        # Compute IoU
        ious = {(image_id, label_id): self._compute_iou(
            y_true=y_true_ddict,
            y_pred=y_pred_ddict,
            image_id=image_id,
            label_id=label_id,
            label_ids=label_ids,
            ) for image_id, label_id in product(images_ids, _label_ids)
            }

        # Evaluate each image
        images_results = [self._evaluate_image(
            y_true=y_true_ddict,
            y_pred=y_pred_ddict,
            image_id=image_id,
            label_id=label_id,
            label_ids=label_ids,
            area_range=_area_range,
            ious=ious
            ) for label_id, _area_range, image_id in
            product(_label_ids, self.area_ranges.values(), images_ids)
        ]

        # Aggregate results
        results = self._aggregate(
            label_ids=_label_ids,
            images_ids=images_ids,
            images_results=images_results,
            )

        # Mean evaluator
        mean_evaluator = partial(
            self._get_mean,
            results=results,
            label_ids=_label_ids
            )

        # Get standard output values (globally)
        standard_results = self._get_standard(
            mean_evaluator=mean_evaluator,
            label_id=None,
            prefix="m",
            )

        # Prepare output
        output: dict[str, float | int | list[int]
                     | dict | np.ndarray | partial] = {}
        output |= standard_results

        # Class metrics
        if self.class_metrics:
            output |= {
                "class_metrics": {
                    label_id: self._get_standard(
                        mean_evaluator=mean_evaluator,
                        label_id=label_id,
                        prefix="",
                        ) for label_id in _label_ids
                        }
                }

        # Add metadata
        output |= {
            "classes": label_ids,
            "n_images": len(images_ids)
            }

        if extended_summary:
            output |= ({k: v for k, v in results.items() if k in ["AP", "AR"]}
                       | {
                           "IoU": ious,
                           "mean_evaluator": mean_evaluator,
                           }
                       )

        return output

    def _compute_iou(
            self,
            y_true: defaultdict,
            y_pred: defaultdict,
            image_id: int,
            label_id: int,
            label_ids: list[int],
            ) -> np.ndarray:
        """
        Compute IoU.

        Parameters
        ----------
        y_true : defaultdict
            Ground truths.
        y_pred : defaultdict
            Predictions.
        image_id : int
            Image id.
        label_id : int
            Label id.
        label_ids : list[int]
            Overall label ids.

        Returns
        -------
        np.ndarray
            `np.ndarray` containing IoU values between `y_true` and `y_pred`.
        """
        if self.class_metrics:
            y_true_ = y_true[image_id, label_id]
            y_pred_ = y_pred[image_id, label_id]
        else:
            y_true_ = reduce(
                lambda x, y: x+y,
                [y_true[image_id, label_id] for label_id in label_ids],
                [],
                )
            y_pred_ = reduce(
                lambda x, y: x+y,
                [y_pred[image_id, label_id] for label_id in label_ids],
                [],
                )

        if not y_pred_ and not y_true_:
            return np.array([])

        # Sort predictions highest score first and cut off to
        # max_detection_thresholds
        y_pred_ = sorted(
            y_pred_,
            key=operator.itemgetter("score"),
            reverse=True
            )[: self.max_detection_thresholds[-1]]

        y_true_boxes = [yt["bbox"] for yt in y_true_]
        y_pred_boxes = [yp["bbox"] for yp in y_pred_]
        iscrowd = [yt["iscrowd"] for yt in y_true_]

        # Compute IoU between each prediction and ground truth region
        ious = iou(
            y_true=y_true_boxes,
            y_pred=y_pred_boxes,
            iscrowd=iscrowd
            )
        return ious

    def _evaluate_image(  # pylint: disable=R0912
            self,
            y_true: defaultdict,
            y_pred: defaultdict,
            image_id: int,
            label_id: int,
            label_ids: list[int],
            area_range: list[float],
            ious: dict,
            ) -> dict[str, Any] | None:
        """
        Evaluate metrics for a single image.

        Parameters
        ----------
        y_true : defaultdict
            Ground truths.
        y_pred : defaultdict
            Predictions.
        image_id : int
            Image id.
        label_id : int
            Label id.
        label_ids : list[int]
            Overall label ids.
        area_range : list[float]
            Area range.
        ious : dict
            IoU dictionary.

        Returns
        -------
        dict[str, Any] | None
            Dictionary containing results given image and label.
            `None` if there is no ground-truths and detections for that
            specific `image_id` and `label_id`.
        """
        if self.class_metrics:
            y_true_ = y_true[image_id, label_id]
            y_pred_ = y_pred[image_id, label_id]
        else:
            y_true_ = reduce(
                lambda x, y: x+y,
                [y_true[image_id, label_id] for label_id in label_ids],
                [],
                )
            y_pred_ = reduce(
                lambda x, y: x+y,
                [y_pred[image_id, label_id] for label_id in label_ids],
                [],
                )
        if not y_true_ and not y_pred_:
            return None

        # Assign _ignore if ignore or outside area range.
        for yt_ in y_true_:
            if (
                    yt_["ignore"] or (yt_["area"] < area_range[0]
                                      or yt_["area"] > area_range[1])
            ):
                yt_["_ignore"] = 1
            else:
                yt_["_ignore"] = 0

        # Sort y_true ignore last
        if len(y_true_) == 0:
            y_true_indexes = ()
        else:
            y_true_indexes, y_true_ = zip(*sorted(
                enumerate(y_true_), key=lambda x: x[1]["_ignore"]))
        iscrowd = [int(yt["iscrowd"]) for yt in y_true_]

        # Sort y_pred highest score first and cut off y_pred to
        # max detection threshold.
        y_pred_ = sorted(y_pred_, key=operator.itemgetter("score"),
                         reverse=True)[: self.max_detection_thresholds[-1]]

        # Load computed ious
        ious = (ious[image_id, label_id][:, y_true_indexes]
                if len(ious[image_id, label_id]) > 0
                else ious[image_id, label_id]
                )

        iou_thresholds_len = len(self.iou_thresholds)
        y_true_len = len(y_true_)
        y_pred_len = len(y_pred_)
        # y_true_matches and y_pred_matches will contain ids of
        # matched prediction and ground truths respectively.
        y_true_matches = np.zeros((iou_thresholds_len, y_true_len))
        y_pred_matches = np.zeros((iou_thresholds_len, y_pred_len))
        y_true_ignore = np.array([yt["_ignore"] for yt in y_true_])
        y_pred_ignore = np.zeros((iou_thresholds_len, y_pred_len))
        if not len(ious) == 0:
            for iou_threshold_index, iou_threshold in enumerate(
                    self.iou_thresholds):
                for yp_index, yp_ in enumerate(y_pred_):
                    iou_ = min([iou_threshold, 1-1e-10])
                    # Information about best match so far
                    # match=-1 -> Unmatched
                    match_ = -1
                    for yt_index, yt_ in enumerate(y_true_):
                        # If this yt already matched, and not a crowd, continue
                        if (y_true_matches[iou_threshold_index, yt_index] > 0
                                and not iscrowd[yt_index]):
                            continue
                        # If yp matched to a previous gt (not ignore) and
                        # the new is ingore, stop
                        if (match_ > -1 and y_true_ignore[match_] == 0
                                and y_true_ignore[yt_index] == 1):
                            break
                        # If iou between yp and yt < iou threshold, continue
                        if ious[yp_index, yt_index] < iou_:
                            continue
                        # If match successful and best so far,
                        # store appropriately
                        iou_ = ious[yp_index, yt_index]
                        match_ = yt_index
                    if match_ == -1:
                        continue
                    # If match made store id of match for both dt and gt
                    y_pred_ignore[iou_threshold_index, yp_index] = (
                        y_true_ignore[match_])
                    y_pred_matches[iou_threshold_index, yp_index] = (
                        y_true_[match_]["id"])
                    y_true_matches[iou_threshold_index, match_] = yp_["id"]
        # set unmatched detections outside of area range to ignore
        out_area = np.array(
            [yp["area"] < area_range[0] or yp["area"] > area_range[1]
             for yp in y_pred_]).reshape((1, len(y_pred_)))
        y_pred_ignore = np.logical_or(
            y_pred_ignore,
            np.logical_and(y_pred_matches == 0,
                           np.repeat(out_area, iou_thresholds_len, 0))
            )
        # store results for given image and label
        return {
                "image_id": image_id,
                "label_id": label_id,
                "area_range": area_range,
                "max_detection_threshold": self.max_detection_thresholds[-1],
                "y_pred_indexes": [yp["id"] for yp in y_pred_],
                "y_true_indexes": [yt["id"] for yt in y_true_],
                "y_pred_matches": y_pred_matches,
                "y_true_matches": y_true_matches,
                "y_pred_scores": [yp["score"] for yp in y_pred_],
                "y_true_ignore": y_true_ignore,
                "y_pred_ignore": y_pred_ignore,
                }

    def _aggregate(  # pylint: disable=R0915
            self,
            label_ids: list[int],
            images_ids: list[int],
            images_results: list[dict[str, Any] | None]
            ) -> dict[str, np.ndarray]:
        """
        Aggregate images results.

        Parameters
        ----------
        label_ids : list[int]
            Overall label ids.
        images_ids : list[int]
            Overall image ids.
        images_results : list[dict[str, Any] | None]
            List of dictionaries containing images results.

        Returns
        -------
        dict[str, np.ndarray]
            Aggregated results.
        """
        # Settings
        iou_thresholds_len = len(self.iou_thresholds)
        recall_thresholds_len = len(self.recall_thresholds)
        label_ids_len = len(label_ids)
        area_range_len = len(self.area_ranges)
        max_detection_thresholds_len = len(self.max_detection_thresholds)
        images_ids_len = len(images_ids)

        # Initialize
        average_precision = -np.ones((
            iou_thresholds_len,
            recall_thresholds_len,
            label_ids_len,
            area_range_len,
            max_detection_thresholds_len
            )
        )
        average_recall = -np.ones((
            iou_thresholds_len,
            label_ids_len,
            area_range_len,
            max_detection_thresholds_len
            )
        )
        scores = -np.ones((
            iou_thresholds_len,
            recall_thresholds_len,
            label_ids_len,
            area_range_len,
            max_detection_thresholds_len
            )
        )

        # Retrieve images_results at each label, area range,
        # and max number of detections
        for label_id_index, _ in enumerate(label_ids):  # pylint: disable=R1702
            n_label = label_id_index*area_range_len*images_ids_len
            for area_range_index, _ in enumerate(self.area_ranges):
                n_area_range = area_range_index*images_ids_len
                for max_det_index, max_det in enumerate(
                        self.max_detection_thresholds):
                    images_results_full = [
                        images_results[n_label + n_area_range + i]
                        for i, _ in enumerate(images_ids)
                        ]
                    images_results_ = [e for e in images_results_full
                                       if e is not None]
                    if len(images_results_) == 0:
                        continue
                    y_pred_scores = np.concatenate(
                        [e["y_pred_scores"][:max_det]
                         for e in images_results_]
                        )

                    if len(y_pred_scores) == 0:
                        indexes = ()
                        y_pred_scores_sorted = y_pred_scores
                    else:
                        indexes, y_pred_scores_sorted = zip(*sorted(
                            enumerate(y_pred_scores),
                            key=lambda x: x[1],
                            reverse=True)
                            )

                    y_pred_matches = np.concatenate(
                        [e["y_pred_matches"][:, :max_det]
                         for e in images_results_],
                        axis=1
                        )[:, indexes]
                    y_pred_ignore = np.concatenate(
                        [e["y_pred_ignore"][:, :max_det]
                         for e in images_results_],
                        axis=1
                        )[:, indexes]
                    y_true_ignore = np.concatenate(
                        [e["y_true_ignore"] for e in images_results_])
                    not_ignore = np.count_nonzero(y_true_ignore == 0)
                    if not_ignore == 0:
                        continue
                    tps = np.logical_and(
                        y_pred_matches,
                        np.logical_not(y_pred_ignore)
                        )
                    fps = np.logical_and(
                        np.logical_not(y_pred_matches),
                        np.logical_not(y_pred_ignore)
                        )

                    tp_sum = np.cumsum(tps, axis=1)\
                        .astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1)\
                        .astype(dtype=float)
                    for iou_threshold_index, (tp_, fp_) in enumerate(
                            zip(tp_sum, fp_sum)):
                        tp_ = np.array(tp_)
                        fp_ = np.array(fp_)
                        n_d = len(tp_)
                        recall = tp_ / not_ignore
                        precision = tp_ / (fp_+tp_+np.spacing(1))
                        qty = np.zeros((recall_thresholds_len,))
                        ss_ = np.zeros((recall_thresholds_len,))

                        if n_d:
                            average_recall[
                                iou_threshold_index,
                                label_id_index,
                                area_range_index,
                                max_det_index
                                ] = recall[-1]
                        else:
                            average_recall[
                                iou_threshold_index,
                                label_id_index,
                                area_range_index,
                                max_det_index
                                ] = 0

                        # Interpolation P-R Curve
                        interpolated_precision = np.maximum.accumulate(
                            precision[::-1])[::-1]

                        indexes_ = np.searchsorted(
                            recall,
                            self.recall_thresholds,
                            side="left"
                            )
                        try:
                            for ri_, pi_ in enumerate(indexes_):
                                qty[ri_] = interpolated_precision[pi_]
                                ss_[ri_] = y_pred_scores_sorted[pi_]
                        except IndexError:
                            pass
                        average_precision[
                            iou_threshold_index,
                            :,
                            label_id_index,
                            area_range_index,
                            max_det_index
                            ] = np.array(qty)
                        scores[
                            iou_threshold_index,
                            :,
                            label_id_index,
                            area_range_index,
                            max_det_index
                            ] = np.array(ss_)
        return {
            "AP": average_precision,
            "AR": average_recall
            }

    def _get_mean(
            self,
            iou_threshold: float | list[float] | np.ndarray | None = None,
            area_range_key: str | list[str] | np.ndarray | None = None,
            max_detection_threshold: (int | list[int] | np.ndarray
                                      | None) = None,
            label_id: int | list[int] | np.ndarray | None = None,
            metrics: (Literal["AP", "AR"] | list[Literal["AP", "AR"]]
                      | None) = None,
            include_spec: bool = False,
            prefix: str = "m",
            results: dict[str, Any] | type[_Missing] = _Missing,
            label_ids: list[int] | np.ndarray | type[_Missing] = _Missing,
            ) -> dict[str, float | dict[str, list]]:
        """
        Calculate mean for Average-Precision (mAP) and Average-Recall (mAR).

        Parameters
        ----------
        iou_threshold : float | list[float] | np.ndarray | None, optional
            IoU threshold on which calculate the mean.
            It can be a `float`, a list of floats, `np.ndarray`
            or `None`; all values must be inlcuded in the
            constructor argument `iou_thresholds`.
            If `None`, all input `iou_thresholds` will be used.
            The default is `None`.
        area_range_key : str | list[str] | np.ndarray | None, optional
            Area range key on which calculate the mean.
            It can be a `str`, a list of strings, `np.ndarray`,
            or `None`; all values must be included in the
            constructor argument `area_ranges`.
            If `None`, all input `area_ranges` keys will be used.
            The default is `None`.
        max_detection_threshold : int | list[int] |
                                   np.ndarray | None), optional
            Threshold on maximum detections per image on
            which calculate the mean.
            It can be a `int`, a list of integers, `np.ndarray`
            or `None`; all values must be inlcuded in the
            constructor argument `max_detection_thresholds`.
            If `None`, all input `max_detection_thresholds`
            will be used.
            The default is `None`.
        label_id : int | list[int] | np.ndarray | None, optional
            Label ids on which calculate the mean.
            If `class_metrics` is `True`, `label_id` must be
            included in the label ids of the provided `y_true`.
            If `class_metrics` is `False`, `label_id` must be `-1`
            (in this case equivalent to `None`).
            If `None`, all labels will be used.
            The default is `None`.
        metrics : Literal["AP", "AR"] | list[Literal["AP", "AR"]] | None,
                  optional
            Metrics on which calculate the mean.
            If `None`, both `"AP"` and `"AR"` will be used.
            The default is `None`.
        include_spec : bool, optional
            Whether to include mean settings specification.
            The default is `False`.
        prefix : str, optional
            Prefix to add to metrics keys.
            The default is `m`.
        results : dict[str, Any] | type[_Missing], optional
            Dictionary containing aggregated images results.
            If `_Missing` an error will be raised.
            The default is `_Missing`.
        label_ids : list[int] | np.ndarray | type[_Missing], optional
            All label ids found in `y_true`.
            If `_Missing` an error will be raised.
            The default is `_Missing`.

        Returns
        -------
        dict[str, float | dict[str, list]]
            Dictionary containing the values of precision and recall.
            If `include_spec` input parameters info will be added.
        """
        # Sanity check
        # results
        if results is _Missing:
            raise TypeError("`results` must be provided.")
        results_ = cast(dict[str, Any], results)
        # label_ids
        if label_ids is _Missing:
            raise TypeError("`label_ids` must be passed.")

        # Default
        default_value = {
            "iou_threshold": self.iou_thresholds,
            "label_id": to_array(label_ids),
            "area_range_key": to_array(list(self.area_ranges.keys())),
            "max_detection_threshold": self.max_detection_thresholds,
            }

        mean_model = MeanModel.model_validate(
            {
                "iou_threshold": iou_threshold,
                "area_range_key": area_range_key,
                "max_detection_threshold": max_detection_threshold,
                "label_id": label_id,
                "metrics": metrics,
                "include_spec": include_spec,
                "prefix": prefix
                },
            context={"default_value": default_value, "default_flag": None}
            )
        metrics = mean_model.metrics
        include_spec = mean_model.include_spec
        prefix = mean_model.prefix

        mean_params = {
            "iou_threshold": mean_model.iou_threshold,
            "label_id": mean_model.label_id,
            "area_range_key": mean_model.area_range_key,
            "max_detection_threshold": mean_model.max_detection_threshold,
            }

        # Indexes
        indexes = {
            key: get_indexes(default_value[key], value)
            for key, value in mean_params.items()
            }

        # Sanity check
        for key, value in mean_params.items():
            if len(value) != len(indexes[key]):
                raise ValueError(
                    f"Input parameter {key}: {value} not found "
                    f"in initial settings which includes {default_value[key]}."
                    )

        # Assigns slice(None) if the indices match the entire dimension
        # of results array.
        indexes_ = {
            key: (
                np.array([slice(None)]) if np.equal(
                    indexes[key],
                    np.arange(len(default_value[key]))
                    ).all() else indexes[key])
            for key in indexes.keys()
        }

        combinations_indexes = [
            dict(zip(indexes_.keys(), values))
            for values in product(*(indexes_[key] for key in indexes_.keys()))
            ]

        output: dict[str, float | dict[str, list]] = {}
        suffix = get_suffix(
            iou_threshold=mean_params["iou_threshold"],
            area_range_key=mean_params["area_range_key"],
            max_detection_threshold=mean_params["max_detection_threshold"]
            )
        for metric in metrics:
            slices: list[tuple]
            if metric == "AP":
                slices = [(
                    index["iou_threshold"],
                    slice(None),
                    index["label_id"],
                    index["area_range_key"],
                    index["max_detection_threshold"],
                    ) for index in combinations_indexes]
            elif metric == "AR":
                slices = [(
                    index["iou_threshold"],
                    index["label_id"],
                    index["area_range_key"],
                    index["max_detection_threshold"],
                    ) for index in combinations_indexes]
            values = np.stack([results_[metric][slice_] for slice_ in slices])
            values = values[values > -1]
            mean_values = -1 if len(values) == 0 else np.mean(values)
            output[f"{prefix}{metric}{suffix}"] = float(mean_values)

        if include_spec:
            output |= {
                "spec": {key: value.tolist()
                         for key, value in mean_params.items()}
                }

        return output

    def _get_standard(
            self,
            mean_evaluator: Callable,
            label_id: int | list[int] | np.ndarray | None,
            prefix: str,
            ) -> dict[str, float | dict[str, list]]:
        """
        Get standard metrics output.

        Parameters
        ----------
        mean_evaluator : Callable
            Mean evaluator function.
        label_id : int | list[int] | np.ndarray | None
            Label ids.
            If `None`, all labels will be used.
            The default is `None`.
        prefix : str
            Prefix to add to metrics keys.

        Returns
        -------
        dict[str, float | dict[str, list]]
            Dictionary with metrics output.
        """
        standard_params = []

        for combination in copy.deepcopy(_STANDARD_OUTPUT):
            if (combination["iou_threshold"] is not None) and (
                    combination["iou_threshold"] not in self.iou_thresholds):
                combination["iou_threshold"] = None

            if combination["area_range_key"] not in self.area_ranges.keys():
                combination["area_range_key"] = None

            if (combination["max_detection_threshold"] not in
                    self.max_detection_thresholds):
                combination["max_detection_threshold"] = None
            standard_params.append(combination)

        output: dict[str, float | dict[str, list]] = {}
        for _param in standard_params:
            output |= mean_evaluator(
                **_param,
                include_spec=False,
                label_id=label_id,
                prefix=prefix,
                )

        output = dict(sorted(output.items()))
        return output


def iou(
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        iscrowd: np.ndarray | list[bool] | list[int] | None = None,
        ) -> np.ndarray:
    """
    Calculate IoU between bounding boxes.

    Single bounding boxes must be in `"xywh"` format, i.e.
        [xmin, ymin, width, height]

    The standard iou of a ground truth `y_true` and detected
    `y_pred` object is:

    $$iou(\\text{y_true}, \\text{y_pred}) =
        \\frac{\\text{y_true} \\bigcap \\text{y_pred}}
        {\\text{y_true}\\bigcup \\text{y_pred}}$$

    Notes
    -----
    For `crowd` regions, COCO use a modified criteria:
        If a `y_true` object is marked as `iscrowd`, it is permissible
        for a detected object `y_pred` to match any subregion of the `y_true`.
        Choosing `y_true'` in the crowd `y_true` that best matches the `y_pred`
        can be done using:
            $$\\text{y_true'} = \\text{y_pred} \\bigcap \\text{y_true}$$
        Since by definition:
            $$ \\text{y_true'} \\bigcup \\text{y_pred} = \\text{y_pred}$$
        computing:
            $$iou(\\text{y_true}, \\text{y_pred}, \\text{iscrowd}) =
               iou(\\text{y_true'}, \\text{y_pred}) =
               \\frac{\\text{y_true} \\bigcap \\text{y_pred}}{\\text{y_pred}}$$
    For crowd regions in ground truth, this modified criteria for IoU
    is applied.

    Raises
    ------
    ValueError
        If `iscrowd` and `y_true` have different length (iscrowd not None).

    Parameters
    ----------
    y_true : np.ndarray | list
        `np.ndarray` with shape `(B1, 4)`, `B1` `y_true` batch size.
    y_pred : np.ndarray | list
        `np.ndarray` with shape `(B2, 4)`, `B2` `y_pred` batch size.
    iscrowd : np.ndarray | list[bool] | list[int] | None
        Whether `y_true` are crowd regions.
        If `None`, it will be set to `False` for all `y_true`.
        The default is `None`.

    Returns
    -------
    np.ndarray
        IoU vector of shape `(B2, B1)`.
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return np.array([])
    # iscrowd
    if iscrowd is not None:
        if len(iscrowd) != len(y_true):
            raise ValueError("`iscrowd` and `y_true` should have the same "
                             "length.")
    else:
        iscrowd = [False]*len(y_true)
    # To np.ndarray
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    # pylint: disable-next=W0632
    xmin1, ymin1, width1, height1 = np.hsplit(y_pred, 4)
    # pylint: disable-next=W0632
    xmin2, ymin2, width2, height2 = np.hsplit(y_true, 4)
    xmax1 = xmin1 + width1
    xmax2 = xmin2 + width2
    ymax1 = ymin1 + height1
    ymax2 = ymin2 + height2

    # Intersection
    xmin_i = np.maximum(xmin1.T, xmin2).T
    ymin_i = np.maximum(ymin1.T, ymin2).T
    xmax_i = np.minimum(xmax1.T, xmax2).T
    ymax_i = np.minimum(ymax1.T, ymax2).T
    inter_area = (np.maximum((xmax_i - xmin_i), 0)
                  * np.maximum((ymax_i - ymin_i), 0))
    # Union
    y_pred_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    y_true_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = y_pred_area + y_true_area.T - inter_area
    det = np.where(
        iscrowd,
        y_pred_area,
        union_area
        )

    result = np.divide(inter_area, det, out=np.zeros(inter_area.shape),
                       where=det != 0)
    return result
