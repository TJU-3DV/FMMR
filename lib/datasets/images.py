"""Image-folder discovery and VGGT preprocessing for the demo."""
from pathlib import Path
from typing import List, Sequence

import numpy as np

from vggt.utils.load_fn import load_and_preprocess_images

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            paths.extend(
                child for child in sorted(path.iterdir())
                if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)
        else:
            raise FileNotFoundError(f"Image or image directory not found: {path}")
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ValueError("No supported image files were found")
    return unique


def select_views(paths: Sequence[Path], max_views: int) -> List[Path]:
    if max_views <= 0 or len(paths) <= max_views:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, max_views, dtype=np.int64)
    return [paths[int(index)] for index in indices]
