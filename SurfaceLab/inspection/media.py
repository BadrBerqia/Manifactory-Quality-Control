"""Lecture des images et rendu des couches pour l'interface."""

import base64
from io import BytesIO

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import SIZE

MAX_UPLOAD = 16 * 1024 * 1024
MAX_PIXELS = 24_000_000
FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}


def read_rgb(data):
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in FORMATS:
                raise ValueError("Formats pris en charge : PNG, JPEG, WebP, BMP et TIFF.")
            if source.width * source.height > MAX_PIXELS or min(source.size) < 16:
                raise ValueError("L’image doit faire au moins 16 × 16 px et au plus 24 mégapixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            image.thumbnail((900, 900), Image.Resampling.LANCZOS)
            return np.asarray(image), (width, height)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        raise ValueError("Impossible de lire cette image. Vérifiez son format et son intégrité.") from None


def truth_mask(path):
    if path is None:
        return np.zeros((SIZE, SIZE), dtype=bool)
    with Image.open(path) as image:
        resized = image.convert("L").resize((SIZE, SIZE), Image.Resampling.NEAREST)
        return np.asarray(resized) > 127


def data_url(image, lossless=False):
    out = BytesIO()
    if lossless:
        Image.fromarray(image).save(out, "PNG")
        mime = "png"
    else:
        Image.fromarray(image).save(out, "JPEG", quality=91, subsampling=0)
        mime = "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(out.getvalue()).decode("ascii")


def render_method(rgb, heat, threshold):
    height, width = rgb.shape[:2]
    size = width, height
    mask = (heat >= threshold).astype(np.uint8)
    display_mask = cv.resize(mask * 255, size, interpolation=cv.INTER_NEAREST)
    heat_display = cv.resize(heat, size, interpolation=cv.INTER_LINEAR)
    colors = cv.applyColorMap(np.uint8(np.clip(heat_display / 2, 0, 1) * 255), cv.COLORMAP_TURBO)
    colors = cv.cvtColor(colors, cv.COLOR_BGR2RGB)
    alpha = np.clip(heat_display * 0.41, 0.10, 0.78)[:, :, None]
    overlay = np.uint8(rgb * (1 - alpha) + colors * alpha)

    # Les boîtes décrivent les composantes du masque dans le repère 256 × 256.
    labels, _, stats, _ = cv.connectedComponentsWithStats(mask)
    contours, _ = cv.findContours(display_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxed = rgb.copy()
    thickness = max(1, round(max(size) / 360))
    cv.drawContours(boxed, contours, -1, (74, 248, 221), thickness)
    for x, y, w, h, _ in stats[1:]:
        first = (round(x * width / SIZE), round(y * height / SIZE))
        last = (round((x + w) * width / SIZE) - 1, round((y + h) * height / SIZE) - 1)
        cv.rectangle(boxed, first, last, (240, 255, 249), thickness)
    return {
        "score": float(heat.max()),
        "regions": labels - 1,
        "area": float(mask.mean()),
        "views": {
            "heat": data_url(overlay),
            "boxes": data_url(boxed),
            "mask": data_url(display_mask, lossless=True),
        },
    }


def render_truth(rgb, truth):
    height, width = rgb.shape[:2]
    mask = cv.resize(truth.astype(np.uint8), (width, height), interpolation=cv.INTER_NEAREST)
    result = rgb.copy()
    region = mask.astype(bool)
    result[region] = np.uint8(0.5 * result[region] + 0.5 * np.array([32, 244, 153]))
    return data_url(result)
