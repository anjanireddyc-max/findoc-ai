"""Error Level Analysis (ELA).

The image is saved again as JPEG at a fixed quality and compared with itself. Areas that
were pasted, re-typed or painted over were compressed a different number of times than the
rest of the page, so their re-compression error differs from their surroundings.
"""
import io

import numpy as np
from PIL import Image

SCALE = 12.0


def ela_map(image, quality=75):
    """Mean absolute re-compression error per pixel (float array, 0-255)."""
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    return np.abs(np.asarray(image, np.float32) - np.asarray(recompressed, np.float32)).mean(axis=2)


def ela_image(image, quality=75):
    """ELA map as an amplified 3-channel image, the input format of the ELA model."""
    e = np.clip(ela_map(image, quality) * SCALE, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([e, e, e], axis=2))
