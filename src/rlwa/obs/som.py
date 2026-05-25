"""Set-of-Mark overlay renderer.

Draws numbered boxes on a screenshot using bounding boxes that BrowserGym
exposes via `extra_element_properties` -> bbox per bid.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_PALETTE = [
    (220, 38, 38), (37, 99, 235), (16, 185, 129), (217, 119, 6),
    (147, 51, 234), (236, 72, 153), (14, 165, 233), (132, 204, 22),
]


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_som(
    screenshot: np.ndarray,
    elements: Dict[str, dict],
    max_marks: int = 40,
) -> Tuple[Image.Image, List[str]]:
    """
    Render Set-of-Mark overlay.

    Args:
        screenshot: HxWx3 uint8 RGB.
        elements:   {bid: {"bbox": [x,y,w,h], "visible": bool, ...}} or similar.
                    Accepts BrowserGym's `extra_element_properties` format.
        max_marks:  cap number of marks (for prompt length / visual clarity).

    Returns:
        (PIL.Image with overlay, list of bids in mark order)
    """
    img = Image.fromarray(screenshot).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _get_font(14)

    # filter visible/interactable elements
    visible = []
    for bid, props in elements.items():
        if not isinstance(props, dict):
            continue
        if props.get("visibility", 1) == 0 or props.get("visible") is False:
            continue
        bbox = props.get("bbox") or props.get("bounding_box_rect")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox[:4]
        if w <= 0 or h <= 0:
            continue
        visible.append((bid, (float(x), float(y), float(w), float(h))))

    visible = visible[:max_marks]
    mark_bids: List[str] = []

    for i, (bid, (x, y, w, h)) in enumerate(visible):
        color = _PALETTE[i % len(_PALETTE)]
        # box
        draw.rectangle([x, y, x + w, y + h], outline=color + (255,), width=2)
        # label background
        label = str(i + 1)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x, max(0, y - th - 4), x + tw + 6, y],
                       fill=color + (230,))
        draw.text((x + 3, max(0, y - th - 3)), label, fill=(255, 255, 255), font=font)
        mark_bids.append(bid)

    return img, mark_bids
