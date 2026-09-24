from __future__ import annotations

from .domain import Rect


def make_head_crop(
    face_box: Rect,
    image_width: int,
    image_height: int,
    margin: float = 2.0,
) -> Rect:
    face_center_x = face_box.x + face_box.w / 2.0
    face_center_y = face_box.y + face_box.h / 2.0

    side = int(max(face_box.w, face_box.h) * margin)
    side = max(side, 64)

    crop_x = int(face_center_x - side / 2.0)
    crop_y = int(face_center_y - side * 0.56)

    crop_x = max(0, min(crop_x, max(0, image_width - side)))
    crop_y = max(0, min(crop_y, max(0, image_height - side)))

    actual_side = min(side, image_width - crop_x, image_height - crop_y)
    return Rect(crop_x, crop_y, actual_side, actual_side)
