"""
Nudity detection wrapper around NudeNet's ONNX model.

We deliberately re-implement the pre/post-processing instead of using
NudeNet's wrapper directly, because the upstream wrapper:

  1. Runs at a hard-coded 320x320 input size (the model file is YOLOv8-nano,
     trained at 320; the ONNX export accepts dynamic shapes so we can run
     bigger and recover small/medium objects the 320x320 pass loses).
  2. Drops every detection below 0.20 score *inside* the postprocessor,
     before any caller can see it (line 98 of nudenet.py).
  3. Applies NMS with a 0.25 score floor (line 126), giving an effective
     floor of 0.25 even if you ask for less.

Our wrapper exposes those knobs so you can decide your own thresholds.

Runs 100% locally; the only network access is the one-time NudeNet model
download to ~/.NudeNet/ (or wherever pip installed the package).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime


# Class label order is fixed by the model. Do not reorder.
MODEL_LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

EXPLICIT_CLASSES = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

SUGGESTIVE_CLASSES = {
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "FEMALE_GENITALIA_COVERED",
    "ANUS_COVERED",
    "BELLY_EXPOSED",
    "ARMPITS_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "FEET_EXPOSED",
}

NEUTRAL_CLASSES = {
    "FACE_FEMALE",
    "FACE_MALE",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
}

CLASS_COLORS = {
    "explicit": (0, 0, 255),
    "suggestive": (0, 165, 255),
    "neutral": (0, 200, 0),
}


def category_for(class_name: str) -> str:
    if class_name in EXPLICIT_CLASSES:
        return "explicit"
    if class_name in SUGGESTIVE_CLASSES:
        return "suggestive"
    return "neutral"


@dataclass
class Detection:
    class_name: str
    score: float
    box: tuple[int, int, int, int]

    @property
    def category(self) -> str:
        return category_for(self.class_name)


@dataclass
class DetectionResult:
    detections: list[Detection]
    image_bgr: np.ndarray

    @property
    def has_explicit(self) -> bool:
        return any(d.category == "explicit" for d in self.detections)

    @property
    def highest_score(self) -> float:
        return max((d.score for d in self.detections), default=0.0)

    def filter(self, min_score: float = 0.0, classes: Iterable[str] | None = None) -> list[Detection]:
        wanted = set(classes) if classes else None
        return [
            d for d in self.detections
            if d.score >= min_score and (wanted is None or d.class_name in wanted)
        ]


def _locate_model() -> str:
    """
    Find the best available NudeNet ONNX file.

    Priority:
      1. ./models_640m.onnx  (the medium model, ~104 MB, much higher accuracy)
      2. The pip package's 320n.onnx (~12 MB, low accuracy fallback)
    """
    project_640m = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_640m.onnx")
    if os.path.isfile(project_640m):
        return project_640m
    import nudenet
    return os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")


def _preprocess(image_bgr: np.ndarray, target_size: int):
    """Letterbox image to a square of `target_size`, return blob + metadata."""
    h, w = image_bgr.shape[:2]
    max_side = max(h, w)
    x_pad = max_side - w
    y_pad = max_side - h
    padded = cv2.copyMakeBorder(image_bgr, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1 / 255.0,
        size=(target_size, target_size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    return blob, x_pad, y_pad, w, h


def _postprocess(
    raw_output: np.ndarray,
    *,
    x_pad: int,
    y_pad: int,
    orig_w: int,
    orig_h: int,
    model_size: int,
    score_threshold: float,
    nms_threshold: float,
    nms_iou: float,
) -> list[Detection]:
    """
    Decode YOLO-style output to Detection objects.

    `raw_output` shape: (1, num_classes+4, num_anchors). After squeeze+transpose:
    (num_anchors, num_classes+4) where columns 0:4 are box (cx, cy, w, h) in
    model-input pixel coords, and columns 4: are per-class scores.
    """
    out = np.transpose(np.squeeze(raw_output))
    boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    sx = (orig_w + x_pad) / model_size
    sy = (orig_h + y_pad) / model_size

    for row in out:
        cls_scores = row[4:]
        max_score = float(np.amax(cls_scores))
        if max_score < score_threshold:
            continue
        cls_id = int(np.argmax(cls_scores))
        cx, cy, bw, bh = row[0:4]

        x = (cx - bw / 2) * sx
        y = (cy - bh / 2) * sy
        w = bw * sx
        h = bh * sy

        x = max(0.0, min(x, orig_w))
        y = max(0.0, min(y, orig_h))
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)

        if w <= 0 or h <= 0:
            continue

        boxes.append([x, y, w, h])
        scores.append(max_score)
        class_ids.append(cls_id)

    if not boxes:
        return []

    # Per class: an EXPOSED box must not be removed because a COVERED box of another
    # class overlaps it with a higher score.
    indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, nms_threshold, nms_iou)
    detections: list[Detection] = []
    for i in np.array(indices).flatten():
        x, y, w, h = boxes[i]
        detections.append(Detection(
            class_name=MODEL_LABELS[class_ids[i]],
            score=float(scores[i]),
            box=(int(x), int(y), int(w), int(h)),
        ))
    return detections


class Detector:
    """
    Direct ONNX-runtime detector over the NudeNet model.

    Args:
        inference_resolution: square input size to run the model at.
            Higher = catches smaller/farther body parts but slower.
            320 = upstream default (fast, misses things).
            640 = recommended balance for moderation.
            1024 = high-recall mode for hard cases.
        score_threshold: minimum class score to keep before NMS.
            Set this LOWER than your final UI threshold so you can see weak
            detections and decide later. Default 0.05 (vs upstream 0.20).
        nms_threshold: minimum score to enter NMS. Default 0.05
            (vs upstream 0.25).
        nms_iou: IoU threshold for NMS overlap suppression. Default 0.45.
    """

    def __init__(
        self,
        *,
        inference_resolution: int = 640,
        score_threshold: float = 0.05,
        nms_threshold: float = 0.05,
        nms_iou: float = 0.45,
        model_path: str | None = None,
    ) -> None:
        self.inference_resolution = inference_resolution
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_iou = nms_iou
        self._session = onnxruntime.InferenceSession(model_path or _locate_model())
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image: str | Path | np.ndarray) -> DetectionResult:
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img_bgr = image

        # cv2.cvtColor RGBA->BGR if needed (matches upstream)
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGBA2BGR)

        blob, x_pad, y_pad, orig_w, orig_h = _preprocess(
            img_bgr, self.inference_resolution
        )
        outputs = self._session.run(None, {self._input_name: blob})
        detections = _postprocess(
            outputs[0],
            x_pad=x_pad,
            y_pad=y_pad,
            orig_w=orig_w,
            orig_h=orig_h,
            model_size=self.inference_resolution,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            nms_iou=self.nms_iou,
        )
        return DetectionResult(detections=detections, image_bgr=img_bgr)


def annotate(
    image_bgr: np.ndarray,
    detections: list[Detection],
    *,
    blur: bool = False,
    blur_categories: set[str] | None = None,
) -> np.ndarray:
    """Return a copy of the image with boxes (and optional blur) drawn on it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    blur_categories = blur_categories or {"explicit", "suggestive"}

    if blur:
        for det in detections:
            if det.category not in blur_categories:
                continue
            x, y, bw, bh = det.box
            x2, y2 = min(x + bw, w), min(y + bh, h)
            x, y = max(0, x), max(0, y)
            if x2 <= x or y2 <= y:
                continue
            roi = out[y:y2, x:x2]
            k = max(15, (min(roi.shape[:2]) // 4) | 1)
            out[y:y2, x:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    for det in detections:
        x, y, bw, bh = det.box
        color = CLASS_COLORS[det.category]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
        label = f"{det.class_name} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y - 6)
        cv2.rectangle(out, (x, ly - th - 4), (x + tw + 4, ly + 2), color, -1)
        cv2.putText(out, label, (x + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out
