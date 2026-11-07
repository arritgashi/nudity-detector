"""
Whole-image NSFW classifier — second opinion to NudeNet.

NudeNet is a body-part *detector* (where in the image is X) trained on a
female-skewed dataset, so it sometimes mislabels male anatomy as female
classes with high confidence. This module adds a complementary
whole-image *classifier* that just answers "is this image NSFW or not"
without trying to localize. The two models together are far more reliable
than either alone.

Default model: Falconsai/nsfw_image_detection
  - ViT-Base, ~340 MB, Apache-2.0
  - Binary output: {"normal": p, "nsfw": q} with p+q ≈ 1
  - Well-tested, no gender bias for the explicit/safe distinction

First call downloads the model from HuggingFace into the local cache
(~/.cache/huggingface/). After that it runs 100% offline on CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import pipeline


DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
# The exact model version that was tested; a new upload to Hugging Face is not taken by itself.
DEFAULT_REVISION = os.environ.get("MOD_CLASSIFIER_REVISION", "96cb0d0342c7afb80cab76ecc58b265fa44da256")


@dataclass
class NSFWScore:
    """Result from the whole-image classifier."""
    nsfw: float            # 0..1 probability the image is NSFW
    normal: float          # 0..1 probability the image is safe
    raw_labels: dict[str, float]  # all labels the model returned, full precision

    @property
    def verdict(self) -> str:
        """Conservative thresholds — tweak in analyzer if needed."""
        if self.nsfw >= 0.7:
            return "NSFW"
        if self.nsfw >= 0.3:
            return "BORDERLINE"
        return "SAFE"


class NSFWClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # device=-1 forces CPU; transformers picks GPU otherwise if available.
        self._pipe = pipeline(
            "image-classification",
            model=model_name,
            revision=DEFAULT_REVISION,
            device=-1,
        )
        self.model_name = model_name

    def classify(self, image: str | Path | Image.Image | np.ndarray) -> NSFWScore:
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR (OpenCV convention) and convert
            if image.ndim == 3 and image.shape[2] == 3:
                rgb = image[:, :, ::-1]
            else:
                rgb = image
            pil = Image.fromarray(rgb)
        else:
            pil = image.convert("RGB")

        raw = self._pipe(pil)
        labels = {item["label"].lower(): float(item["score"]) for item in raw}
        # A model whose labels changed would score every image 0 (safe). Fail instead.
        if not labels.keys() & {"nsfw", "porn", "explicit", "hentai", "sexy"}:
            raise RuntimeError(f"Unexpected classifier labels: {sorted(labels)}")

        # Falconsai returns "nsfw" / "normal". Be tolerant of other label names.
        nsfw = labels.get("nsfw") or labels.get("porn") or labels.get("explicit") or 0.0
        normal = labels.get("normal") or labels.get("safe") or labels.get("sfw") or 0.0

        # If model has more granular labels (e.g. drawing/hentai/sexy/porn),
        # treat anything sexual as nsfw.
        for key, val in labels.items():
            if key in {"hentai", "sexy", "porn", "explicit"}:
                nsfw = max(nsfw, val)
            if key in {"neutral", "drawing"}:
                normal = max(normal, val)

        return NSFWScore(nsfw=nsfw, normal=normal, raw_labels=labels)
