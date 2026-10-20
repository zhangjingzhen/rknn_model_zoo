#!/usr/bin/env python3
"""Run YOLOv8 with the RKNN-Toolkit2 PC simulator.

This program intentionally does not connect to a Rockchip device.  The ONNX
model is built and initialized in the same process because an exported RKNN
model cannot be loaded back into the PC simulator for inference.
"""

import argparse
import sys
from pathlib import Path

import cv2
from rknn.api import RKNN


SCRIPT_DIR = Path(__file__).resolve().parent
YOLOV8_DIR = SCRIPT_DIR.parent
REPO_ROOT = YOLOV8_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from py_utils.coco_utils import COCO_test_helper  # noqa: E402
from yolov8 import IMG_SIZE, draw, post_process  # noqa: E402


DEFAULT_MODEL = YOLOV8_DIR / "model" / "yolov8n.onnx"
DEFAULT_IMAGE = YOLOV8_DIR / "model" / "bus.jpg"
DEFAULT_DATASET = REPO_ROOT / "datasets" / "COCO" / "coco_subset_20.txt"
DEFAULT_OUTPUT = YOLOV8_DIR / "result_simulator.jpg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 inference with the local RKNN-Toolkit2 simulator"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="optimized YOLOv8 ONNX model")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="input image")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="annotated output image")
    parser.add_argument("--platform", default="rk3588",
                        help="target platform used while building the model")
    parser.add_argument("--quantize", action="store_true",
                        help="build an INT8 model using the calibration dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="quantization calibration dataset list")
    parser.add_argument("--export-rknn", type=Path,
                        help="optionally export the built model; inference still uses the simulator")
    parser.add_argument("--verbose", action="store_true",
                        help="show detailed RKNN-Toolkit2 logs")
    return parser.parse_args()


def require_file(path, description):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def check_ret(ret, action):
    if ret != 0:
        raise RuntimeError(f"{action} failed with RKNN error code {ret}")


def preprocess(image):
    helper = COCO_test_helper(enable_letter_box=True)
    resized = helper.letter_box(
        im=image.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(0, 0, 0),
    )
    # RKNN config handles division by 255. Keep a uint8 NHWC RGB input here
    # and add the batch dimension expected by inference().
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb[None, ...], helper


def main():
    args = parse_args()
    model_path = require_file(args.model, "ONNX model")
    image_path = require_file(args.image, "input image")
    dataset_path = None
    if args.quantize:
        dataset_path = require_file(args.dataset, "quantization dataset")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {image_path}")
    input_data, letterbox_helper = preprocess(image)

    rknn = RKNN(verbose=args.verbose)
    try:
        print(f"[1/6] Configuring model for {args.platform}")
        check_ret(
            rknn.config(
                mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=args.platform,
            ),
            "config",
        )

        print(f"[2/6] Loading ONNX model: {model_path}")
        check_ret(rknn.load_onnx(model=str(model_path)), "load_onnx")

        mode = "INT8" if args.quantize else "FP16"
        print(f"[3/6] Building {mode} RKNN model")
        check_ret(
            rknn.build(
                do_quantization=args.quantize,
                dataset=str(dataset_path) if dataset_path else None,
            ),
            "build",
        )

        if args.export_rknn:
            export_path = args.export_rknn.expanduser().resolve()
            export_path.parent.mkdir(parents=True, exist_ok=True)
            check_ret(rknn.export_rknn(str(export_path)), "export_rknn")
            print(f"      Exported model: {export_path}")

        print("[4/6] Initializing local simulator (no target/device_id)")
        # Do not pass target or device_id: that would request connected-device inference.
        check_ret(rknn.init_runtime(), "init_runtime")

        print(f"[5/6] Running inference: {image_path}")
        outputs = rknn.inference(inputs=[input_data], data_format="nhwc")
        if outputs is None:
            raise RuntimeError("inference returned no outputs")
        boxes, classes, scores = post_process(outputs)

        result = image.copy()
        if boxes is None:
            print("No objects detected")
        else:
            boxes = letterbox_helper.get_real_box(boxes)
            draw(result, boxes, scores, classes)
            print(f"Detected {len(boxes)} object(s)")

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"failed to save result: {output_path}")
        print(f"[6/6] Result saved to: {output_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
