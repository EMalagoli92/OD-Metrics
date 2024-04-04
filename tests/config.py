"""Unittest configurations."""

__all__ = [
    "TESTS",
    ]

import numpy as np
from pydantic import ValidationError


iou_thresholds_tests = [
    {
     "metrics_settings": {
         "iou_thresholds": [.67],
         },
     "compute_settings": {"extended_summary": True},
     "ids": "iou_thresholds_list",
     },
    {
     "metrics_settings": {
         "iou_thresholds": np.array([.67]),
     },
     "compute_settings": {"extended_summary": True},
     "ids": "iou_thresholds_np.ndarray"
     },
    {
     "metrics_settings": {
         "iou_thresholds": .67,
     },
     "compute_settings": {"extended_summary": True},
     "ids": "iou_thresholds_float"
     },
    {
     "metrics_settings": {
         "iou_thresholds": ".67",
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "iou_thresholds_ValidationError1"
     },
    {
     "metrics_settings": {
         "iou_thresholds": None,
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "iou_thresholds_ValidationError2"
     },
    {
     "metrics_settings": {
         "iou_thresholds": [.5, .5],
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "iou_thresholds_ValidationError3"
     },
    ]


recall_thresholds_tests = [
    {
     "metrics_settings": {
         "recall_thresholds": [.2, .3, .4, .5],
         },
     "compute_settings": {"extended_summary": True},
     "ids": "recall_thresholds_list"
     },
    {
     "metrics_settings": {
         "recall_thresholds": np.array([.2, .3, .4, .5]),
     },
     "compute_settings": {"extended_summary": True},
     "ids": "recall_thresholds_np.ndarray"
     },
    {
     "metrics_settings": {
         "recall_thresholds": .2,
     },
     "compute_settings": {"extended_summary": True},
     "ids": "recall_thresholds_float"
     },
    {
     "metrics_settings": {
         "recall_thresholds": ".2",
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "recall_thresholds_ValidationError1"
     },
    {
     "metrics_settings": {
         "recall_thresholds": None,
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "recall_thresholds_ValidationError2"
     },
    {
     "metrics_settings": {
         "recall_thresholds": [.2, .2],
     },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "recall_thresholds_ValidationError3"
     },
    ]


max_detection_thresholds_tests = [
    {
     "metrics_settings": {
         "max_detection_thresholds": [2],
         },
     "compute_settings": {"extended_summary": True},
     "ids": "max_detection_thresholds_list_int"
     },
    {
     "metrics_settings": {
         "max_detection_thresholds": 2,
         },
     "compute_settings": {"extended_summary": True},
     "ids": "max_detection_thresholds_int"
     },
    {
     "metrics_settings": {
         "max_detection_thresholds": np.array([2]),
         },
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
        "y_true": {"min_objects": 3},
        "y_pred": {"min_objects": 3},
        },
     "ids": "max_detection_thresholds_np.ndarray"
     },
    {
     "metrics_settings": {
         "max_detection_thresholds": None,
         },
     "compute_settings": {"extended_summary": True},
     "ids": "max_detection_thresholds_None"
     },
    {
     "metrics_settings": {
         "max_detection_thresholds": "2",
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "max_detection_thresholds_ValidationError1"
     },
    {
     "metrics_settings": {
         "max_detection_thresholds": [2, 2],
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "max_detection_thresholds_ValidationError2"
     },
    ]


area_ranges_tests = [
    {
     "metrics_settings": {
         "area_ranges": {"custom1": [10, 10000]},
         },
     "compute_settings": {"extended_summary": True},
     "ids": "area_ranges_list_float"
     },
    {
     "metrics_settings": {
         "area_ranges": {"custom2": np.array([10, 10000])},
         },
     "compute_settings": {"extended_summary": True},
     "ids": "area_ranges_np.ndarray"
     },
    {
     "metrics_settings": {
         "area_ranges": None,
         },
     "compute_settings": {"extended_summary": True},
     "ids": "area_ranges_None"
     },
    {
     "metrics_settings": {
         "area_ranges": 10,
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError1"
     },
    {
     "metrics_settings": {
         "area_ranges": {4: [3, 2]},
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError2"
     },
    {
     "metrics_settings": {
         "area_ranges": {"custom3": [3, 2, 4]},
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError3"
     },
    {
     "metrics_settings": {
         "area_ranges": {"custom3": [3, "a"]},
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError4"
     },
    {
     "metrics_settings": {
         "area_ranges": {"custom3": [3, 2]},
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError5"
     },
    {
     "metrics_settings": {
         "area_ranges": {"custom1": [2, 3],
                         "custom2": [2, 3]},
         },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "area_ranges_ValidationError6"
     },
    ]


box_format_tests = [
    {
     "metrics_settings": {
        "box_format": "cxcywh"
        },
     "compute_settings": {"extended_summary": True},
     "ids": "box_format_cxcy"
    },
    {
     "metrics_settings": {
        "box_format": "xyxy"
        },
     "compute_settings": {"extended_summary": True},
     "ids": "box_format_xyxy"
     },
    {
     "metrics_settings": {
        "box_format": "xywh"
        },
     "compute_settings": {"extended_summary": True},
     "ids": "box_format_xywh"
     },
    {
     "metrics_settings": {
        "box_format": "new"
        },
     "compute_settings": {"extended_summary": True},
     "exceptions": {"init": ValidationError},
     "ids": "box_format_ValueError"
     }
    ]


objects_number_tests = [
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {
             "max_objects": 0,
             },
         "y_pred": {},
         },
     "ids": "objects_number_y_true_no_objects"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {},
         "y_pred": {
             "max_objects": 0,
             },
         },
     "ids": "objects_number_y_pred_no_objects"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {
             "max_objects": 0,
             },
         "y_pred": {
             "max_objects": 0,
             },
         },
     "ids": "objects_number_y_pred_y_true_no_objects"
     },
    ]


objects_size_tests = [
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {
             "max_box_width": 0,
             },
         "y_pred": {},
         },
     "ids": "objects_size_y_true_0_width"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {
             "max_box_height": 0,
             },
         "y_pred": {},
            },
     "ids": "objects_size_y_true_0_height"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {
             "max_box_height": 0,
             "max_box_width": 0
             },
         "y_pred": {},
             },
     "ids": "objects_size_y_true_0_height_0_width"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {},
         "y_pred": {
             "max_box_width": 0
             },
         },
     "ids": "objects_size_y_pred_0_width"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {},
         "y_pred": {
             "max_box_height": 0
             },
         },
     "ids": "objects_size_y_pred_0_height"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {},
         "y_pred": {
             "max_box_height": 0,
             "max_box_width": 0
             },
         },
     "ids": "objects_size_y_pred_0_height_0_width"
     }
    ]


mean_evaluator_tests = [
    # iou_threshold
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": .5,
         },
     "ids": "mean_evaluator_iou_float"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": [.5],
         },
     "ids": "mean_evaluator_iou_list_float"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": np.array([.5]),
         },
     "ids": "mean_evaluator_iou_list_float"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": None,
         },
     "ids": "mean_evaluator_iou_None"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": [.4],
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_iou_ValueError"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "iou_threshold": [.5, .5],
         },
     "exceptions": {"mean_evaluator": ValidationError},
     "ids": "mean_evaluator_iou_ValidationError"
     },
    # area_range_key
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": "all",
         },
     "ids": "mean_evaluator_area_range_key_str"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": ["all"],
         },
     "ids": "mean_evaluator_area_range_key_list_str"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": ["all"],
         },
     "ids": "mean_evaluator_area_range_key_np.array"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": None,
         },
     "ids": "mean_evaluator_area_range_key_None"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": "error",
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_area_range_key_ValueError"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "area_range_key": ["all", "all"],
         },
     "exceptions": {"mean_evaluator": ValidationError},
     "ids": "mean_evaluator_area_range_key_ValidationError"
     },
    # max_detection_threshold
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": 10,
         },
     "ids": "mean_evaluator_max_detection_int"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": [10],
         },
     "ids": "mean_evaluator_max_detection_list_int"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": np.array([10]),
         },
     "ids": "mean_evaluator_max_detection_np.array"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": None,
         },
     "ids": "mean_evaluator_max_detection_None"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": 13,
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_max_detection_ValueError"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "max_detection_threshold": [10, 10],
         },
     "exceptions": {"mean_evaluator": ValidationError},
     "ids": "mean_evaluator_max_detection_ValidationError"
     },
    # label_id
    {
     "metrics_settings": {
         "class_metrics": [True],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": 2,
         },
     "ids": "mean_evaluator_label_id_int1"
     },
    {
     "metrics_settings": {
         "class_metrics": [False],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": None,
         },
     "ids": "mean_evaluator_label_id_None"
     },
    {
     "metrics_settings": {
         "class_metrics": [False],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": -1,
         },
     "ids": "mean_evaluator_label_id_-1"
     },
    {
     "metrics_settings": {
         "class_metrics": [True],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": np.array([2]),
         },
     "ids": "mean_evaluator_label_id_np.array"
     },
    {
     "metrics_settings": {
         "class_metrics": [True],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": 3,
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_ValueError"
     },
    {
     "metrics_settings": {
         "class_metrics": [False],
         },
     "annotations_settings": {
         "y_true": {
             "n_classes": 3,
             },
         "y_pred": {
             "n_classes": 3,
             },
         },
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "label_id": 2,
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_ValueError2"
     },
    # metrics
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "metrics": "AP",
         },
     "ids": "mean_evaluator_metrics_str"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "metrics": ["AP", "AR"],
         },
     "ids": "mean_evaluator_metrics_list"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "metrics": None,
         },
     "ids": "mean_evaluator_metrics_None"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "metrics": ["metric"],
         },
     "exceptions": {"mean_evaluator": ValueError},
     "ids": "mean_evaluator_metrics_ValueError"
     },
    # include_spec
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "include_spec": True,
         },
     "ids": "mean_evaluator_include_spec_True"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "include_spec": False,
         },
     "ids": "mean_evaluator_include_spec_False"
     },
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "include_spec": "spec",
         },
     "exceptions": {"mean_evaluator": ValidationError},
     "ids": "mean_evaluator_include_spec_ValueError"
     },
    # prefix
    {
     "compute_settings": {"extended_summary": True},
     "mean_evaluator_settings": {
         "prefix": 2,
         },
     "exceptions": {"mean_evaluator": ValidationError},
     "ids": "mean_evaluator_include_spec_ValueError"
     },
    ]

misc_tests = [
    {
     "compute_settings": {"extended_summary": True},
     "ids": "default_COCO",
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {"n_classes": 3},
         "y_pred": {"n_classes": 7},
         },
     "ids": "misc_default_COCO_different_classes_y_true_y_pred"
     },
    {
     "compute_settings": {"extended_summary": True},
     "annotations_settings": {
         "y_true": {"n_images": 10},
         "y_pred": {"n_images": 5},
         },
     "exceptions": {"compute": ValidationError},
     "ids": "misc_exception_compute_different_images"
     },
    {
     "compute_settings": {"extended_summary": "yes"},
     "exceptions": {"compute": ValidationError},
     "ids": "misc_exception_extended_summary"
     },
    ]


tests: list[dict] = (
    iou_thresholds_tests
    + recall_thresholds_tests
    + max_detection_thresholds_tests
    + area_ranges_tests
    + box_format_tests
    + objects_number_tests
    + objects_size_tests
    + mean_evaluator_tests
    + misc_tests
    )

TESTS = []
for test in tests:
    class_metrics = test.get("metrics_settings",
                             {}).get("class_metrics", [True, False])
    for class_metrics_ in class_metrics:
        test_tmp = test.copy()
        test_tmp["metrics_settings"] = test_tmp.get(
            "metrics_settings",
            {}
            ) | {"class_metrics": class_metrics_}
        test_tmp["compute_settings"] = test_tmp.get(
            "compute_settings",
            {}
            )
        test_tmp["annotations_settings"] = test_tmp.get(
            "annotations_settings",
            {
                "y_true": {},
                "y_pred": {},
                }
            )
        test_tmp["mean_evaluator_settings"] = test_tmp.get(
            "mean_evaluator_settings",
            {},
            )
        test_tmp["ids"] += f"__classs_metrics_{class_metrics_}"
        test_tmp["exceptions"] = test_tmp.get(
            "exceptions",
            {}
            )
        TESTS.append(test_tmp)
