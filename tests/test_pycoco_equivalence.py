"""Unittest pycocotools equivalence."""

from __future__ import annotations

import unittest
import copy
from typing import Any
import numpy as np
from parameterized import parameterized_class

from src.od_metrics import ODMetrics, iou
from src.od_metrics.constants import DEFAULT_COCO
from tests.utils import annotations_generator, pycoco_converter,\
    test_equality, rename_dict, xywh_to
from tests.config import TESTS

try:
    import pycocotools
    from pycocotools import mask as maskUtils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    _PYCOCOTOOLS_AVAILABLE = False

# pylint: disable=E1101


@unittest.skipUnless(
    _PYCOCOTOOLS_AVAILABLE,
    "This unittest needs `pycocotools`. Please intall by "
    "running `pip install pycocotools`"
    )
@parameterized_class(TESTS)
class TestPyCocoEquivalenceMetrics(unittest.TestCase):
    """Test equivalence: ODMetrics and pycocotools.COCOeval"""

    metrics_settings: dict
    compute_settings: dict
    annotations_settings: dict
    mean_evaluator_settings: dict
    exceptions: dict

    def get_pycoco_params(
            self,
            od_metrics_params: dict[str, Any],
            pycoco_params: pycocotools.cocoeval.Params,
            real_max_detections: int | None,
            ) -> pycocotools.cocoeval.Params:
        """Set pycocotools.cocoeval.Params."""
        pycoco_params.iouThrs = od_metrics_params["iou_thresholds"]
        pycoco_params.recThrs = od_metrics_params["recall_thresholds"]
        # Handle max_detection_thresholds case
        pycoco_params.maxDets = (
            od_metrics_params["max_detection_thresholds"]
            if not np.equal(
                    od_metrics_params["max_detection_thresholds"],
                    np.array([None]),
                    ).all()
            else [real_max_detections + 1]
            )
        pycoco_params.areaRng = list(od_metrics_params["area_ranges"].values())
        pycoco_params.areaRngLbl = list(
            od_metrics_params["area_ranges"].keys())
        pycoco_params.useCats = int(od_metrics_params["class_metrics"])

        return pycoco_params

    @staticmethod
    def _test_ious(
            od_metrics_ious: dict,
            pycoco_ious: dict,
            ) -> bool:
        """Test equivalence: od-metrics ious and pycocotools.ious."""
        od_metrics_ious = dict(map(
            lambda x: (x[0], [] if np.array_equal(x[1], np.array([]))
                       else x[1]), od_metrics_ious.items()
            )
        )
        return test_equality(
            od_metrics_ious,
            pycoco_ious,
            )

    @staticmethod
    def _test_aggregate(
            od_metrics_output: dict,
            pycoco_eval: dict,
            ) -> bool:
        """Test equivalence: od-metrics and pycocotools aggregated results."""
        od_metrics_output = copy.deepcopy(od_metrics_output)
        pycoco_eval = copy.deepcopy(pycoco_eval)

        _keys = ["precision", "recall"]
        od_metrics_output = rename_dict(
            dict_=od_metrics_output,
            rename_dict_keys={"AP": "precision",
                              "AR": "recall"}
            )
        return test_equality(
            {k: od_metrics_output[k] for k in _keys},
            {k: pycoco_eval[k] for k in _keys}
            )

    @staticmethod
    def _test_summary(
            od_metrics_output: dict,
            pycoco_stats: np.ndarray,
            is_default_coco: bool,
            ) -> bool:
        """Test equivalence: od-metrics summary and pycocotools summary."""
        # Check default settings
        if is_default_coco:
            od_metrics_stats = np.array([
                od_metrics_output["mAP@[.5:.95 | all | 100]"],
                od_metrics_output["mAP@[.5 | all | 100]"],
                od_metrics_output["mAP@[.75 | all | 100]"],
                od_metrics_output["mAP@[.5:.95 | small | 100]"],
                od_metrics_output["mAP@[.5:.95 | medium | 100]"],
                od_metrics_output["mAP@[.5:.95 | large | 100]"],
                od_metrics_output["mAR@[.5:.95 | all | 1]"],
                od_metrics_output["mAR@[.5:.95 | all | 10]"],
                od_metrics_output["mAR@[.5:.95 | all | 100]"],
                od_metrics_output["mAR@[.5:.95 | small | 100]"],
                od_metrics_output["mAR@[.5:.95 | medium | 100]"],
                od_metrics_output["mAR@[.5:.95 | large | 100]"]
                ]
            )

            return test_equality(
                pycoco_stats,
                od_metrics_stats,
                )
        return True

    # pylint: disable=R0915, R0912, R0914
    def test_equivalence(self) -> None:
        """Test equivalence: od-metrics and pycocotools."""
        # Get annotations
        y_true_od_metrics = annotations_generator(
            **self.annotations_settings["y_true"])
        y_pred_od_metrics = annotations_generator(
            **self.annotations_settings["y_pred"], include_score=True)
        # max detections: Only used for max_detections_thresholds=None case
        real_max_detections = (
            max(detect["boxes"].shape[0] for detect in y_pred_od_metrics)
            if "max_detection_thresholds" in self.metrics_settings and
               self.metrics_settings["max_detection_thresholds"] is None
            else None
        )

        # Prepare pycoco annotations
        y_true_pycoco = pycoco_converter(y_true_od_metrics)
        y_pred_pycoco = pycoco_converter(y_pred_od_metrics)

        # Run Od_metrics evaluation
        # Init
        _init_exception = self.exceptions.get("init", None)
        if _init_exception is None:
            od_metrics_obj = ODMetrics(**self.metrics_settings)
        else:
            with self.assertRaises(_init_exception):
                od_metrics_obj = ODMetrics(**self.metrics_settings)
            return
        # Box format
        y_true_od_metrics = [
            ann | {
                "boxes": [
                    list(xywh_to(box, od_metrics_obj.box_format))
                    for box in ann["boxes"]]} for ann in y_true_od_metrics
            ]
        y_pred_od_metrics = [
            ann | {
                "boxes": [
                    list(xywh_to(box, od_metrics_obj.box_format))
                    for box in ann["boxes"]]} for ann in y_pred_od_metrics
            ]

        # Compute
        _compute_exception = self.exceptions.get("compute", None)
        if _compute_exception is None:
            od_metrics_output = od_metrics_obj.compute(
                y_true=y_true_od_metrics,
                y_pred=y_pred_od_metrics,
                **self.compute_settings,
                )
        else:
            with self.assertRaises(_compute_exception):
                od_metrics_output = od_metrics_obj.compute(
                    y_true=y_true_od_metrics,
                    y_pred=y_pred_od_metrics,
                    **self.compute_settings,
                    )
            return

        # Check default coco
        is_default_coco = self.is_default_coco(od_metrics_obj)

        # Run pycoco evaluation
        pycoco_target, pycoco_preds = COCO(), COCO()
        pycoco_target.dataset = y_true_pycoco
        pycoco_preds.dataset = y_pred_pycoco
        pycoco_target.createIndex()
        pycoco_preds.createIndex()
        pycoco_obj = COCOeval(pycoco_target, pycoco_preds, "bbox")
        params = self.get_pycoco_params(
            od_metrics_obj.__dict__,
            pycoco_obj.params,
            real_max_detections=real_max_detections
            )
        pycoco_obj.params = params
        pycoco_obj.evaluate()
        pycoco_obj.accumulate()
        if is_default_coco:
            pycoco_obj.summarize()

        # Test IoUs equivalence
        _iou_excpetion = self.exceptions.get("iou", None)
        if _iou_excpetion is None:
            with self.subTest("Test IoU"):
                self.assertTrue(self._test_ious(
                    od_metrics_ious=od_metrics_output["IoU"],
                    pycoco_ious=pycoco_obj.ious
                    )
                )
        else:
            with self.assertRaises(_iou_excpetion):
                with self.subTest("Test IoU"):
                    self._test_ious(
                        od_metrics_ious=od_metrics_output["IoU"],
                        pycoco_ious=pycoco_obj.ious
                        )
            return

        # Test aggregate equivalence
        _aggregate_exception = self.exceptions.get("aggregate", None)
        if _aggregate_exception is None:
            with self.subTest("Test aggregate"):
                self.assertTrue(self._test_aggregate(
                    od_metrics_output=od_metrics_output,
                    pycoco_eval=pycoco_obj.eval
                    )
                )
        else:
            with self.assertRaises(_aggregate_exception):
                self._test_aggregate(
                    od_metrics_output=od_metrics_output,
                    pycoco_eval=pycoco_obj.eval
                    )
            return

        # Test summary equivalence
        _summarize_exception = self.exceptions.get("summarize", None)
        if _summarize_exception is None:
            with self.subTest("Test summarize"):
                self.assertTrue(self._test_summary(
                    od_metrics_output=od_metrics_output,
                    pycoco_stats=pycoco_obj.stats,
                    is_default_coco=is_default_coco,
                    )
                )
        else:
            with self.assertRaises(_summarize_exception):
                self._test_summary(
                    od_metrics_output=od_metrics_output,
                    pycoco_stats=pycoco_obj.stats,
                    is_default_coco=is_default_coco,
                    )
            return

        # Test mean evalautor
        _mean_evaluator_exception = self.exceptions.get("mean_evaluator", None)
        if _mean_evaluator_exception is None:
            with self.subTest("Test mean evaluator"):
                od_metrics_output["mean_evaluator"](
                    **self.mean_evaluator_settings)
        else:
            with self.assertRaises(_mean_evaluator_exception):
                od_metrics_output["mean_evaluator"](
                    **self.mean_evaluator_settings)
            return

    @staticmethod
    def is_default_coco(od_metrics_obj) -> bool:
        """Check if settings are default COCO ones."""
        return (
            (np.array_equal(od_metrics_obj.iou_thresholds,
                            DEFAULT_COCO["iou_thresholds"]))
            and (np.array_equal(od_metrics_obj.recall_thresholds,
                                DEFAULT_COCO["recall_thresholds"]))
            and (np.array_equal(od_metrics_obj.max_detection_thresholds,
                                DEFAULT_COCO["max_detection_thresholds"]))
            and (od_metrics_obj.area_ranges
                 == DEFAULT_COCO["area_ranges"])
        )


@unittest.skipUnless(
    _PYCOCOTOOLS_AVAILABLE,
    "This unittest needs `pycocotools`. Please intall by "
    "running `pip install pycocotools`"
    )
class TestPyCocoEquivalenceIoU(unittest.TestCase):
    """Test equivalence: od-metrics and pycocotools iou function."""

    HIGH = 100000
    SIZE = 3000

    def test_iou(self) -> None:
        """Test IoU."""
        iscrowd = list(map(
            bool,
            np.random.randint(
                low=0,
                high=2,
                size=[self.SIZE]
                ).tolist()
            )
        )
        y_pred = np.random.randint(
            low=1,
            high=self.HIGH,
            size=[self.SIZE, 4]
            )
        y_true = np.random.randint(
            low=1,
            high=self.HIGH,
            size=[self.SIZE, 4]
            )

        od_metrics_ious = iou(
            y_true=y_true,
            y_pred=y_pred,
            iscrowd=iscrowd
            )
        pycoco_ious = maskUtils.iou(y_pred, y_true, iscrowd)

        self.assertTrue(test_equality(od_metrics_ious, pycoco_ious))


if __name__ == "__main__":
    unittest.main()
