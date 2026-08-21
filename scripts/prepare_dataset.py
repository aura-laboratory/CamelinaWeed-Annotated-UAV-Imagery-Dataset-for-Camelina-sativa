#!/usr/bin/env python3
"""
CamelinaWeed unified detection and instance-segmentation experiment pipeline.

What this script supports
-------------------------
1. Select any acquisition folders with --include.
2. Split data:
   - random image-level split (exploratory only),
   - flight/folder-wise split,
   - field-wise split,
   - season-wise split,
   - manual split through JSON.
3. Sample weed-negative TRAINING source images from each selected acquisition
   at a configurable ratio (e.g. 0.10 or 0.15 relative to positive images).
   Validation and test keep all available negatives.
4. Convert expert polygons into axis-aligned detection boxes.
5. Tile images only AFTER the train/val/test assignment.
6. Train and evaluate:
   - Ultralytics YOLO detection,
   - lightweight Ultralytics YOLO detection,
   - Torchvision Faster R-CNN,
   - Ultralytics RT-DETR,
   - Ultralytics YOLO instance segmentation.
7. Report broadleaf/narrowleaf:
   - Precision, Recall, F1,
   - AP@50,
   - AP@50:95,
   - macro averages,
   - per-flight/field/season metrics.
8. Repeat experiments over several seeds and export mean ± standard deviation.

Important scientific note
-------------------------
For the revised paper, use a flight-wise, field-wise, season-wise, or manual
held-out split. Random image-level splitting is included only for exploratory
comparisons because overlapping UAV frames can cause data leakage.

Expected dataset layout
-----------------------
CamelinaWeed/
  Summer 2025/
    Thessaloniki/
      Phantom flight at 5 m altitude/
        Annotated/
          images/
          annotations.json
        Unannotated/
          images/
  Winter 2025-2026/
    ...

Dependencies
------------
pip install numpy pillow tqdm torch torchvision ultralytics

Examples
--------
A. Inspect available acquisition folders:
python prepare_dataset.py list-folders --input "D:/CamelinaWeed"

B. Prepare a manual leakage-safe split:
python prepare_dataset.py prepare ^
  --input "D:/CamelinaWeed" ^
  --output "D:/CamelinaPrepared" ^
  --split-strategy manual ^
  --manual-split split_config.json ^
  --class-map class_map.json ^
  --negative-ratio 0.15 ^
  --tile-size 768 ^
  --tile-overlap 0.20

C. Train all requested detectors with 3 seeds:
python prepare_dataset.py train ^
  --dataset "D:/CamelinaPrepared" ^
  --output "D:/CamelinaResults" ^
  --models yolo yolo_light fasterrcnn rtdetr ^
  --seeds 42 52 62 ^
  --group-metrics flight

D. Prepare a YOLO instance-segmentation dataset:
python prepare_dataset.py prepare ^
  --input "D:/CamelinaWeed" ^
  --output "D:/CamelinaPreparedSeg" ^
  --task segmentation ^
  --split-strategy manual ^
  --manual-split split_config.json ^
  --class-map class_map.json ^
  --negative-ratio 0.15 ^
  --tile-size 768

E. Train the segmentation baseline:
python prepare_dataset.py train ^
  --dataset "D:/CamelinaPreparedSeg" ^
  --output "D:/CamelinaSegResults" ^
  --models yolo_seg ^
  --seg-weights yolo26n-seg.pt ^
  --seeds 42 52 62

F. Run preparation and training in one command:
python prepare_dataset.py all [all prepare arguments] [all train arguments]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants and data classes
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

CLASS_NAMES = ["broadleaf", "narrowleaf"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class ObjectAnnotation:
    class_id: int
    box_xyxy: tuple[float, float, float, float]
    # One exterior polygon in source-image pixel coordinates. COCO annotations
    # with multiple polygon components are represented by their largest valid
    # component because one YOLO segmentation row describes one object polygon.
    polygon_xy: list[tuple[float, float]] | None = None


@dataclass
class SourceImage:
    path: Path
    season: str
    location: str
    acquisition: str
    flight_key: str
    field_key: str
    season_key: str
    annotations: list[ObjectAnnotation]
    is_negative_source: bool
    split: str | None = None


@dataclass
class TileRecord:
    tile_path: Path
    label_path: Path
    split: str
    season: str
    location: str
    acquisition: str
    flight: str
    field: str
    source_image: str
    is_negative_source: bool
    object_count: int


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Normalize Unicode dashes, slashes, case, and repeated whitespace."""
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\\", "/")
    for dash in ("–", "—", "−"):
        value = value.replace(dash, "-")
    value = " ".join(value.strip().lower().split())
    value = value.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Reproducibility is preferred over maximum speed for reported runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_torch_device(device_value: str):
    """Resolve Ultralytics-style device values safely for Torchvision."""
    import torch

    value = str(device_value).strip().lower()
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        return torch.device("mps")
    if value.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{value}")
        print(
            "WARNING: CUDA was not found; Faster R-CNN will run on CPU.",
            file=sys.stderr,
        )
        return torch.device("cpu")
    return torch.device(value)


def parse_negative_ratio(value: str) -> float | str:
    value = value.strip().lower()
    if value == "all":
        return "all"
    ratio = float(value)
    if not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError(
            "negative ratio must be between 0 and 1, or 'all'"
        )
    return ratio


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p.resolve()
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_annotation_json(annotated_dir: Path) -> Path:
    candidates = [
        annotated_dir / "_annotations.coco.json",
        annotated_dir / "annotations.json",
        annotated_dir / "_annotations.json",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    json_files = sorted(annotated_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0].resolve()
    if not json_files:
        raise FileNotFoundError(f"No COCO JSON found in {annotated_dir}")
    raise RuntimeError(
        f"Multiple JSON files found in {annotated_dir}; "
        "rename the intended file to annotations.json"
    )


def find_coco_image(annotated_dir: Path, file_name: str) -> Path:
    candidates = [
        annotated_dir / file_name,
        annotated_dir / "images" / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Image '{file_name}' listed in COCO JSON was not found in {annotated_dir}"
    )


def is_multispectral_acquisition(name: str) -> bool:
    """Avoid confusing the '3M' in DJI Mavic 3M with an MS acquisition."""
    text = normalize_text(name)
    return (
        text.endswith(" altitude ms")
        or text.endswith(" flight ms")
        or "multispectral" in text
        or text.endswith("/ms")
    )


# ---------------------------------------------------------------------------
# Folder discovery and class mapping
# ---------------------------------------------------------------------------

def discover_acquisitions(
    dataset_root: Path,
    include_patterns: Sequence[str],
    modality: str,
) -> list[Path]:
    """
    Return acquisition folders, where each folder contains an Annotated child.

    An acquisition path is expected to be:
        root / season / location / acquisition
    """
    normalized_patterns = [normalize_text(p) for p in include_patterns]
    acquisitions: list[Path] = []

    for annotated_dir in dataset_root.rglob("*"):
        if not annotated_dir.is_dir():
            continue
        if normalize_text(annotated_dir.name) != "annotated":
            continue

        acquisition_dir = annotated_dir.parent.resolve()
        relative = normalize_text(
            acquisition_dir.relative_to(dataset_root.resolve()).as_posix()
        )

        if normalized_patterns and not any(p in relative for p in normalized_patterns):
            continue

        if modality == "rgb" and is_multispectral_acquisition(acquisition_dir.name):
            continue
        if modality == "ms" and not is_multispectral_acquisition(acquisition_dir.name):
            continue

        acquisitions.append(acquisition_dir)

    acquisitions = sorted(set(acquisitions))
    if not acquisitions:
        raise RuntimeError(
            "No acquisition folders were found. Check --input, --include, "
            "the modality, and the Annotated folder names."
        )
    return acquisitions


def load_class_map(path: Path | None) -> dict[str, str]:
    """
    Load a mapping from original COCO labels to broadleaf/narrowleaf.

    Matching is case-insensitive. Example:
    {
      "Broadleaf": "broadleaf",
      "Narrowleaf": "narrowleaf",
      "Sinapis arvensis": "broadleaf",
      "Avena sterilis": "narrowleaf"
    }
    """
    if path is None:
        return {
            normalize_text("broadleaf"): "broadleaf",
            normalize_text("narrowleaf"): "narrowleaf",
        }

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict) or not raw:
        raise ValueError("class-map JSON must be a non-empty object")

    mapping: dict[str, str] = {}
    for original, target in raw.items():
        target_norm = normalize_text(str(target))
        if target_norm not in CLASS_TO_ID:
            raise ValueError(
                f"Invalid target class '{target}'. Use broadleaf or narrowleaf."
            )
        mapping[normalize_text(str(original))] = target_norm
    return mapping


def resolve_target_class(
    category_name: str,
    class_map: dict[str, str],
) -> int | None:
    normalized = normalize_text(category_name)

    if normalized in class_map:
        return CLASS_TO_ID[class_map[normalized]]

    # Safe automatic fallback only for explicit category terms.
    if "broadleaf" in normalized:
        return CLASS_TO_ID["broadleaf"]
    if "narrowleaf" in normalized:
        return CLASS_TO_ID["narrowleaf"]

    return None


# ---------------------------------------------------------------------------
# COCO parsing and polygon-to-box conversion
# ---------------------------------------------------------------------------

def valid_polygons(
    annotation: dict[str, Any],
) -> list[list[tuple[float, float]]]:
    """Return valid COCO polygon components as pixel-coordinate point lists."""
    segmentations = annotation.get("segmentation", [])
    polygons: list[list[tuple[float, float]]] = []

    if not isinstance(segmentations, list):
        return polygons

    if segmentations and all(isinstance(v, (int, float)) for v in segmentations):
        segmentations = [segmentations]

    for segment in segmentations:
        if (
            not isinstance(segment, list)
            or len(segment) < 6
            or len(segment) % 2 != 0
        ):
            continue

        polygon = [
            (float(segment[idx]), float(segment[idx + 1]))
            for idx in range(0, len(segment), 2)
        ]

        # Remove a repeated closing point, because YOLO does not require it.
        if len(polygon) >= 2 and polygon[0] == polygon[-1]:
            polygon = polygon[:-1]

        if len({(round(x, 6), round(y, 6)) for x, y in polygon}) >= 3:
            polygons.append(polygon)

    return polygons


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Return the absolute shoelace area of a polygon."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def largest_valid_polygon(
    annotation: dict[str, Any],
) -> list[tuple[float, float]] | None:
    """
    Return the largest valid exterior polygon for one COCO annotation.

    Most CamelinaWeed instances contain one polygon component. If an annotation
    contains several disjoint components, YOLO's one-row-per-object polygon
    format cannot preserve them as one instance, so the largest component is
    used and the choice is deterministic.
    """
    polygons = [
        polygon
        for polygon in valid_polygons(annotation)
        if polygon_area(polygon) > 0.0
    ]
    return max(polygons, key=polygon_area) if polygons else None


def valid_polygon_vertices(annotation: dict[str, Any]) -> list[tuple[float, float]]:
    """Flatten valid polygon components for detection-box calculation."""
    return [
        point
        for polygon in valid_polygons(annotation)
        for point in polygon
    ]


def polygon_or_coco_box(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    Prefer a box computed from polygon extrema. Fall back to COCO bbox.

    Returns x1, y1, x2, y2 in pixel coordinates.
    """
    vertices = valid_polygon_vertices(annotation)

    if vertices:
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(float(image_width), max(xs))
        y2 = min(float(image_height), max(ys))
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    bbox = annotation.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, width, height = map(float, bbox)
        x1 = max(0.0, min(x, float(image_width)))
        y1 = max(0.0, min(y, float(image_height)))
        x2 = max(x1, min(x + width, float(image_width)))
        y2 = max(y1, min(y + height, float(image_height)))
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    return None


def load_positive_images_from_acquisition(
    dataset_root: Path,
    acquisition_dir: Path,
    class_map: dict[str, str],
    ignore_unmapped: bool,
) -> tuple[list[SourceImage], set[str]]:
    annotated_dir = acquisition_dir / "Annotated"
    json_path = find_annotation_json(annotated_dir)

    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    images = {
        int(image["id"]): image
        for image in coco.get("images", [])
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    relative = acquisition_dir.relative_to(dataset_root).parts
    if len(relative) < 3:
        raise ValueError(
            f"Expected season/location/acquisition hierarchy, got: {acquisition_dir}"
        )

    season = relative[0]
    location = relative[1]
    acquisition = "/".join(relative[2:])
    flight_key = f"{season}/{location}/{acquisition}"
    field_key = f"{season}/{location}"
    season_key = season

    source_images: list[SourceImage] = []
    unmapped: set[str] = set()

    for image_id, image_info in images.items():
        image_path = find_coco_image(annotated_dir, str(image_info["file_name"]))
        width = int(image_info["width"])
        height = int(image_info["height"])
        objects: list[ObjectAnnotation] = []

        for annotation in annotations_by_image.get(image_id, []):
            category_name = categories.get(
                int(annotation["category_id"]),
                f"class_{annotation['category_id']}",
            )
            target_class = resolve_target_class(category_name, class_map)
            if target_class is None:
                unmapped.add(category_name)
                if ignore_unmapped:
                    continue
                continue

            box = polygon_or_coco_box(annotation, width, height)
            polygon = largest_valid_polygon(annotation)
            if box is not None:
                objects.append(
                    ObjectAnnotation(
                        class_id=target_class,
                        box_xyxy=box,
                        polygon_xy=polygon,
                    )
                )

        # Annotated files without a relevant object are retained as background.
        source_images.append(
            SourceImage(
                path=image_path,
                season=season,
                location=location,
                acquisition=acquisition,
                flight_key=flight_key,
                field_key=field_key,
                season_key=season_key,
                annotations=objects,
                is_negative_source=(len(objects) == 0),
            )
        )

    return source_images, unmapped


def load_unannotated_images_from_acquisition(
    dataset_root: Path,
    acquisition_dir: Path,
) -> list[SourceImage]:
    relative = acquisition_dir.relative_to(dataset_root).parts
    season = relative[0]
    location = relative[1]
    acquisition = "/".join(relative[2:])
    flight_key = f"{season}/{location}/{acquisition}"
    field_key = f"{season}/{location}"

    unannotated_dir = acquisition_dir / "Unannotated"
    paths = iter_images(unannotated_dir)

    return [
        SourceImage(
            path=path,
            season=season,
            location=location,
            acquisition=acquisition,
            flight_key=flight_key,
            field_key=field_key,
            season_key=season,
            annotations=[],
            is_negative_source=True,
        )
        for path in paths
    ]


def collect_source_images(
    dataset_root: Path,
    acquisitions: Sequence[Path],
    class_map: dict[str, str],
    ignore_unmapped: bool,
) -> list[SourceImage]:
    all_images: list[SourceImage] = []
    unmapped_all: set[str] = set()

    for acquisition in acquisitions:
        annotated_images, unmapped = load_positive_images_from_acquisition(
            dataset_root,
            acquisition,
            class_map,
            ignore_unmapped,
        )
        unannotated_images = load_unannotated_images_from_acquisition(
            dataset_root,
            acquisition,
        )

        all_images.extend(annotated_images)
        all_images.extend(unannotated_images)
        unmapped_all.update(unmapped)

    if unmapped_all and not ignore_unmapped:
        labels = "\n".join(f"  - {name}" for name in sorted(unmapped_all))
        raise ValueError(
            "The following annotation labels are not mapped to broadleaf or "
            f"narrowleaf:\n{labels}\n"
            "Add them to --class-map, or use --ignore-unmapped only if they "
            "should genuinely be excluded."
        )

    if not all_images:
        raise RuntimeError("No source images were collected.")

    return all_images


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def source_group_key(image: SourceImage, strategy: str) -> str:
    if strategy in {"folder", "flight"}:
        return image.flight_key
    if strategy == "field":
        return image.field_key
    if strategy == "season":
        return image.season_key
    raise ValueError(f"Unsupported grouped split strategy: {strategy}")


def proportional_boundaries(
    count: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[int, int]:
    train_end = int(round(count * train_ratio))
    val_end = train_end + int(round(count * val_ratio))

    # Keep all subsets non-empty where possible.
    if count >= 3:
        train_end = min(max(train_end, 1), count - 2)
        val_end = min(max(val_end, train_end + 1), count - 1)

    return train_end, val_end


def split_random_images(
    images: list[SourceImage],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    train_end, val_end = proportional_boundaries(
        len(shuffled), train_ratio, val_ratio
    )

    for index, image in enumerate(shuffled):
        if index < train_end:
            image.split = "train"
        elif index < val_end:
            image.split = "val"
        else:
            image.split = "test"


def split_grouped(
    images: list[SourceImage],
    strategy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    groups = sorted({source_group_key(image, strategy) for image in images})
    if len(groups) < 3:
        raise ValueError(
            f"The '{strategy}' strategy found only {len(groups)} groups. "
            "At least three are needed for automatic train/val/test assignment. "
            "Use --split-strategy manual for a scientifically meaningful split."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)
    train_end, val_end = proportional_boundaries(
        len(groups), train_ratio, val_ratio
    )

    assignments = {}
    for index, group in enumerate(groups):
        if index < train_end:
            assignments[group] = "train"
        elif index < val_end:
            assignments[group] = "val"
        else:
            assignments[group] = "test"

    for image in images:
        image.split = assignments[source_group_key(image, strategy)]


def load_manual_split(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if set(data) != {"train", "val", "test"}:
        raise ValueError(
            "Manual split JSON must contain exactly train, val, and test."
        )

    result = {}
    for split, patterns in data.items():
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"Manual split '{split}' must be a non-empty list.")
        result[split] = [normalize_text(str(pattern)) for pattern in patterns]
    return result


def split_manual(
    images: list[SourceImage],
    manual_split_path: Path,
) -> None:
    rules = load_manual_split(manual_split_path)

    for image in images:
        key = normalize_text(image.flight_key)
        matches = [
            split
            for split, patterns in rules.items()
            if any(pattern in key for pattern in patterns)
        ]

        if len(set(matches)) != 1:
            raise ValueError(
                f"Manual split for '{image.flight_key}' matched {matches}. "
                "Each acquisition must match exactly one subset."
            )
        image.split = matches[0]


def apply_split(
    images: list[SourceImage],
    strategy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    manual_split_path: Path | None,
) -> None:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Train and validation ratios must be positive and sum to < 1.")

    if strategy == "random":
        split_random_images(images, train_ratio, val_ratio, seed)
    elif strategy in {"folder", "flight", "field", "season"}:
        split_grouped(images, strategy, train_ratio, val_ratio, seed)
    elif strategy == "manual":
        if manual_split_path is None:
            raise ValueError("--manual-split is required for manual strategy.")
        split_manual(images, manual_split_path)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    if any(image.split is None for image in images):
        raise RuntimeError("Some source images were not assigned to a split.")


def audit_group_leakage(
    images: Sequence[SourceImage],
    group_by: str,
) -> None:
    """
    Audit leakage at the requested level.

    Even when random splitting is selected, the audit warns rather than fails,
    because random mode is intentionally available for exploratory comparison.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for image in images:
        if group_by == "flight":
            key = image.flight_key
        elif group_by == "field":
            key = image.field_key
        elif group_by == "season":
            key = image.season_key
        else:
            return
        owners[key].add(str(image.split))

    leaked = {key: splits for key, splits in owners.items() if len(splits) > 1}
    if leaked:
        details = "\n".join(
            f"  - {key}: {sorted(splits)}"
            for key, splits in sorted(leaked.items())
        )
        print(
            "\nWARNING: group leakage was detected:\n"
            f"{details}\n"
            "Do not use this split as the main revised-paper experiment.\n",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Negative-image sampling
# ---------------------------------------------------------------------------

def sample_training_negatives(
    images: list[SourceImage],
    negative_ratio: float | str,
    seed: int,
) -> list[SourceImage]:
    """
    Apply the negative ratio only to TRAINING source images.

    Ratio definition:
        selected training negatives per acquisition
        = ceil(number of positive training source images * negative_ratio)

    All validation and test negatives are retained for robust false-positive
    evaluation.
    """
    rng = random.Random(seed)
    kept: list[SourceImage] = [
        image for image in images
        if image.split != "train" or not image.is_negative_source
    ]

    train_by_flight: dict[str, list[SourceImage]] = defaultdict(list)
    for image in images:
        if image.split == "train" and image.is_negative_source:
            train_by_flight[image.flight_key].append(image)

    positives_by_flight = Counter(
        image.flight_key
        for image in images
        if image.split == "train" and not image.is_negative_source
    )

    for flight, candidates in sorted(train_by_flight.items()):
        rng.shuffle(candidates)

        if negative_ratio == "all":
            selected_count = len(candidates)
        else:
            selected_count = math.ceil(
                positives_by_flight.get(flight, 0) * float(negative_ratio)
            )
            selected_count = min(selected_count, len(candidates))

        selected = candidates[:selected_count]
        kept.extend(selected)

        print(
            f"[negative sampling] {flight}: "
            f"{positives_by_flight.get(flight, 0)} positive train images, "
            f"{len(candidates)} available negatives, "
            f"{len(selected)} selected."
        )

    return kept


# ---------------------------------------------------------------------------
# Tiling and YOLO export
# ---------------------------------------------------------------------------

def tile_positions(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def intersection_box(
    box: tuple[float, float, float, float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    ix1 = max(x1, tile_x)
    iy1 = max(y1, tile_y)
    ix2 = min(x2, tile_x + tile_size)
    iy2 = min(y2, tile_y + tile_size)

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def clipped_annotations_for_tile(
    annotations: Sequence[ObjectAnnotation],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
) -> list[ObjectAnnotation]:
    clipped: list[ObjectAnnotation] = []

    for annotation in annotations:
        original = annotation.box_xyxy
        intersection = intersection_box(original, tile_x, tile_y, tile_size)
        if intersection is None:
            continue

        x1, y1, x2, y2 = original
        ix1, iy1, ix2, iy2 = intersection
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        visible_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

        if original_area <= 0 or visible_area / original_area < min_visibility:
            continue

        clipped.append(
            ObjectAnnotation(
                class_id=annotation.class_id,
                box_xyxy=(
                    ix1 - tile_x,
                    iy1 - tile_y,
                    ix2 - tile_x,
                    iy2 - tile_y,
                ),
            )
        )

    return clipped



def _clip_polygon_boundary(
    points: list[tuple[float, float]],
    inside,
    intersection,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping against one rectangular boundary."""
    if not points:
        return []

    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = inside(previous)

    for current in points:
        current_inside = inside(current)

        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))

        previous = current
        previous_inside = current_inside

    return output


def _vertical_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    boundary_x: float,
) -> tuple[float, float]:
    x1, y1 = first
    x2, y2 = second
    if abs(x2 - x1) < 1e-12:
        return boundary_x, y1
    ratio = (boundary_x - x1) / (x2 - x1)
    return boundary_x, y1 + ratio * (y2 - y1)


def _horizontal_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    boundary_y: float,
) -> tuple[float, float]:
    x1, y1 = first
    x2, y2 = second
    if abs(y2 - y1) < 1e-12:
        return x1, boundary_y
    ratio = (boundary_y - y1) / (y2 - y1)
    return x1 + ratio * (x2 - x1), boundary_y


def _remove_duplicate_polygon_points(
    points: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        point = (float(x), float(y))
        if not cleaned or (
            abs(point[0] - cleaned[-1][0]) > 1e-8
            or abs(point[1] - cleaned[-1][1]) > 1e-8
        ):
            cleaned.append(point)

    if len(cleaned) >= 2 and (
        abs(cleaned[0][0] - cleaned[-1][0]) <= 1e-8
        and abs(cleaned[0][1] - cleaned[-1][1]) <= 1e-8
    ):
        cleaned.pop()

    return cleaned


def clip_polygon_to_rectangle(
    polygon: Sequence[tuple[float, float]],
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle."""
    points = list(polygon)

    points = _clip_polygon_boundary(
        points,
        inside=lambda point: point[0] >= x_min,
        intersection=lambda first, second: _vertical_intersection(
            first, second, x_min
        ),
    )
    points = _clip_polygon_boundary(
        points,
        inside=lambda point: point[0] <= x_max,
        intersection=lambda first, second: _vertical_intersection(
            first, second, x_max
        ),
    )
    points = _clip_polygon_boundary(
        points,
        inside=lambda point: point[1] >= y_min,
        intersection=lambda first, second: _horizontal_intersection(
            first, second, y_min
        ),
    )
    points = _clip_polygon_boundary(
        points,
        inside=lambda point: point[1] <= y_max,
        intersection=lambda first, second: _horizontal_intersection(
            first, second, y_max
        ),
    )

    return _remove_duplicate_polygon_points(points)


def clipped_segmentation_annotations_for_tile(
    annotations: Sequence[ObjectAnnotation],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_visibility: float,
) -> list[ObjectAnnotation]:
    """
    Clip source polygons to one tile and convert them to tile-local coordinates.

    Objects without a valid COCO polygon are excluded from segmentation export.
    """
    clipped_annotations: list[ObjectAnnotation] = []
    x_max = tile_x + tile_size
    y_max = tile_y + tile_size

    for annotation in annotations:
        if not annotation.polygon_xy:
            continue

        source_area = polygon_area(annotation.polygon_xy)
        if source_area <= 0.0:
            continue

        clipped_global = clip_polygon_to_rectangle(
            annotation.polygon_xy,
            float(tile_x),
            float(tile_y),
            float(x_max),
            float(y_max),
        )
        clipped_area = polygon_area(clipped_global)

        if (
            len(clipped_global) < 3
            or clipped_area <= 1.0
            or clipped_area / source_area < min_visibility
        ):
            continue

        local_polygon = [
            (
                min(max(x - tile_x, 0.0), float(tile_size)),
                min(max(y - tile_y, 0.0), float(tile_size)),
            )
            for x, y in clipped_global
        ]
        local_polygon = _remove_duplicate_polygon_points(local_polygon)

        if len(local_polygon) < 3 or polygon_area(local_polygon) <= 1.0:
            continue

        xs = [point[0] for point in local_polygon]
        ys = [point[1] for point in local_polygon]
        clipped_annotations.append(
            ObjectAnnotation(
                class_id=annotation.class_id,
                box_xyxy=(min(xs), min(ys), max(xs), max(ys)),
                polygon_xy=local_polygon,
            )
        )

    return clipped_annotations

def save_yolo_labels(
    label_path: Path,
    annotations: Sequence[ObjectAnnotation],
    tile_size: int,
    task: str,
) -> None:
    """
    Save YOLO detection or instance-segmentation labels.

    Detection row:
        class_id x_center y_center width height

    Segmentation row:
        class_id x1 y1 x2 y2 ... xn yn
    """
    lines: list[str] = []

    for annotation in annotations:
        if task == "detection":
            x1, y1, x2, y2 = annotation.box_xyxy
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2.0
            y_center = y1 + height / 2.0

            values = [
                x_center / tile_size,
                y_center / tile_size,
                width / tile_size,
                height / tile_size,
            ]
            lines.append(
                f"{annotation.class_id} "
                + " ".join(f"{min(max(value, 0.0), 1.0):.6f}" for value in values)
            )

        elif task == "segmentation":
            if not annotation.polygon_xy or len(annotation.polygon_xy) < 3:
                continue

            normalized: list[float] = []
            for x, y in annotation.polygon_xy:
                normalized.extend(
                    [
                        min(max(x / tile_size, 0.0), 1.0),
                        min(max(y / tile_size, 0.0), 1.0),
                    ]
                )

            lines.append(
                f"{annotation.class_id} "
                + " ".join(f"{value:.6f}" for value in normalized)
            )

        else:
            raise ValueError(f"Unsupported preparation task: {task}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def tile_one_source_image(
    source: SourceImage,
    output_root: Path,
    tile_size: int,
    overlap: float,
    min_visibility: float,
    negative_tiles_per_train_image: int,
    seed: int,
    task: str,
) -> list[TileRecord]:
    image = open_rgb_image(source.path)
    width, height = image.size

    x_positions = tile_positions(width, tile_size, overlap)
    y_positions = tile_positions(height, tile_size, overlap)
    positions = [(x, y) for y in y_positions for x in x_positions]

    stable_digest = hashlib.sha1(
        str(source.path.resolve()).encode("utf-8")
    ).hexdigest()
    stable_offset = int(stable_digest[:10], 16) % 1_000_000
    rng = random.Random(seed + stable_offset)

    # Training negative source images contribute only a controlled number of
    # random background tiles. Validation/test retain all tiles.
    if (
        source.split == "train"
        and source.is_negative_source
        and negative_tiles_per_train_image > 0
        and len(positions) > negative_tiles_per_train_image
    ):
        positions = rng.sample(positions, negative_tiles_per_train_image)

    records: list[TileRecord] = []

    for tile_index, (tile_x, tile_y) in enumerate(positions):
        if task == "detection":
            clipped = clipped_annotations_for_tile(
                source.annotations,
                tile_x,
                tile_y,
                tile_size,
                min_visibility,
            )
        elif task == "segmentation":
            clipped = clipped_segmentation_annotations_for_tile(
                source.annotations,
                tile_x,
                tile_y,
                tile_size,
                min_visibility,
            )
        else:
            raise ValueError(f"Unsupported preparation task: {task}")

        # For positive TRAINING source images, discard empty tiles.
        # Validation/test preserve the complete tiled scene.
        if (
            source.split == "train"
            and not source.is_negative_source
            and not clipped
        ):
            continue

        crop = image.crop(
            (
                tile_x,
                tile_y,
                min(tile_x + tile_size, width),
                min(tile_y + tile_size, height),
            )
        )
        if crop.size != (tile_size, tile_size):
            crop = ImageOps.pad(
                crop,
                (tile_size, tile_size),
                color=(0, 0, 0),
                centering=(0.0, 0.0),
            )

        safe_source = "".join(
            character if character.isalnum() else "_"
            for character in source.path.stem
        )
        source_token = hashlib.sha1(
            str(source.path.resolve()).encode("utf-8")
        ).hexdigest()[:10]
        tile_name = (
            f"{source_token}_{safe_source}"
            f"__x{tile_x}_y{tile_y}_t{tile_index:04d}.jpg"
        )

        image_dir = output_root / "images" / str(source.split)
        label_dir = output_root / "labels" / str(source.split)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        tile_path = image_dir / tile_name
        label_path = label_dir / f"{Path(tile_name).stem}.txt"

        crop.save(tile_path, quality=95)
        save_yolo_labels(label_path, clipped, tile_size, task)

        records.append(
            TileRecord(
                tile_path=tile_path,
                label_path=label_path,
                split=str(source.split),
                season=source.season,
                location=source.location,
                acquisition=source.acquisition,
                flight=source.flight_key,
                field=source.field_key,
                source_image=str(source.path),
                is_negative_source=source.is_negative_source,
                object_count=len(clipped),
            )
        )

    return records


def write_dataset_yaml(output_root: Path) -> None:
    yaml_text = (
        f"path: {output_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: broadleaf\n"
        "  1: narrowleaf\n"
    )
    (output_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_tile_manifest(output_root: Path, records: Sequence[TileRecord]) -> None:
    manifest_path = output_root / "tile_manifest.csv"
    fields = [
        "tile_path",
        "label_path",
        "split",
        "season",
        "location",
        "acquisition",
        "flight",
        "field",
        "source_image",
        "is_negative_source",
        "object_count",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "tile_path": str(record.tile_path.resolve()),
                    "label_path": str(record.label_path.resolve()),
                    "split": record.split,
                    "season": record.season,
                    "location": record.location,
                    "acquisition": record.acquisition,
                    "flight": record.flight,
                    "field": record.field,
                    "source_image": record.source_image,
                    "is_negative_source": record.is_negative_source,
                    "object_count": record.object_count,
                }
            )


def write_source_manifest(output_root: Path, images: Sequence[SourceImage]) -> None:
    fields = [
        "source_image",
        "split",
        "season",
        "location",
        "acquisition",
        "flight",
        "field",
        "is_negative_source",
        "object_count",
    ]
    with (output_root / "source_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for image in sorted(images, key=lambda item: str(item.path)):
            writer.writerow(
                {
                    "source_image": str(image.path.resolve()),
                    "split": image.split,
                    "season": image.season,
                    "location": image.location,
                    "acquisition": image.acquisition,
                    "flight": image.flight_key,
                    "field": image.field_key,
                    "is_negative_source": image.is_negative_source,
                    "object_count": len(image.annotations),
                }
            )


def prepare_dataset(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    dataset_root = args.input.resolve()
    output_root = args.output.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Use --overwrite to recreate it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    acquisitions = discover_acquisitions(
        dataset_root,
        args.include,
        args.modality,
    )

    print("\nSelected acquisition folders:")
    for acquisition in acquisitions:
        print(f"  - {acquisition.relative_to(dataset_root)}")

    class_map = load_class_map(args.class_map)
    images = collect_source_images(
        dataset_root,
        acquisitions,
        class_map,
        args.ignore_unmapped,
    )

    apply_split(
        images,
        args.split_strategy,
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.manual_split,
    )

    audit_group_leakage(images, "flight")

    images = sample_training_negatives(
        images,
        args.negative_ratio,
        args.seed,
    )

    if args.task == "segmentation":
        total_objects = sum(len(image.annotations) for image in images)
        polygon_objects = sum(
            annotation.polygon_xy is not None
            for image in images
            for annotation in image.annotations
        )
        if polygon_objects == 0:
            raise ValueError(
                "Segmentation preparation was requested, but no valid COCO "
                "polygons were found."
            )
        if polygon_objects < total_objects:
            print(
                "WARNING: "
                f"{total_objects - polygon_objects} annotations have no valid "
                "polygon and will be excluded from segmentation labels.",
                file=sys.stderr,
            )

    print_source_summary(images)

    records: list[TileRecord] = []
    for source in tqdm(images, desc="Tiling source images"):
        records.extend(
            tile_one_source_image(
                source=source,
                output_root=output_root,
                tile_size=args.tile_size,
                overlap=args.tile_overlap,
                min_visibility=args.min_visibility,
                negative_tiles_per_train_image=args.negative_tiles_per_image,
                seed=args.seed,
                task=args.task,
            )
        )

    write_dataset_yaml(output_root)
    write_tile_manifest(output_root, records)
    write_source_manifest(output_root, images)
    print_tile_summary(records)

    settings = {
        "input": str(dataset_root),
        "selected_acquisitions": [
            str(path.relative_to(dataset_root)) for path in acquisitions
        ],
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "negative_ratio_train_only": args.negative_ratio,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "min_visibility": args.min_visibility,
        "negative_tiles_per_train_image": args.negative_tiles_per_image,
        "class_names": CLASS_NAMES,
        "task": args.task,
    }
    (output_root / "preparation_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    print(f"\nPrepared dataset saved to: {output_root}")


def print_source_summary(images: Sequence[SourceImage]) -> None:
    print("\nSource-image split summary")
    print("=" * 84)
    for split in ("train", "val", "test"):
        subset = [image for image in images if image.split == split]
        positives = sum(not image.is_negative_source for image in subset)
        negatives = sum(image.is_negative_source for image in subset)
        objects = sum(len(image.annotations) for image in subset)
        print(
            f"{split:>5}: {len(subset):5d} images | "
            f"{positives:5d} positive | {negatives:5d} negative | "
            f"{objects:7d} objects"
        )
    print("=" * 84)


def print_tile_summary(records: Sequence[TileRecord]) -> None:
    print("\nTile split summary")
    print("=" * 84)
    for split in ("train", "val", "test"):
        subset = [record for record in records if record.split == split]
        positive = sum(record.object_count > 0 for record in subset)
        empty = len(subset) - positive
        objects = sum(record.object_count for record in subset)
        print(
            f"{split:>5}: {len(subset):6d} tiles | "
            f"{positive:6d} with objects | {empty:6d} empty | "
            f"{objects:8d} objects"
        )
    print("=" * 84)


# ---------------------------------------------------------------------------
# Prepared YOLO dataset reader
# ---------------------------------------------------------------------------

def read_yolo_label(
    label_path: Path,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    labels: list[int] = []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    for line in text.splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO detection line in {label_path}: {line}")

        class_id = int(values[0])
        xc, yc, box_w, box_h = map(float, values[1:])
        xc *= width
        yc *= height
        box_w *= width
        box_h *= height

        boxes.append(
            [
                xc - box_w / 2.0,
                yc - box_h / 2.0,
                xc + box_w / 2.0,
                yc + box_h / 2.0,
            ]
        )
        labels.append(class_id)

    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_manifest(dataset_root: Path) -> dict[str, dict[str, str]]:
    manifest_path = dataset_root / "tile_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    result = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[Path(row["tile_path"]).name] = row
    return result


def split_image_paths(dataset_root: Path, split: str) -> list[Path]:
    image_dir = dataset_root / "images" / split
    return sorted(
        path
        for path in image_dir.glob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ground_truth_for_split(
    dataset_root: Path,
    split: str,
) -> dict[str, dict[str, np.ndarray]]:
    truth = {}
    for image_path in split_image_paths(dataset_root, split):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
        boxes, labels = read_yolo_label(label_path, width, height)
        truth[image_path.name] = {"boxes": boxes, "labels": labels}
    return truth


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def box_iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = box_area + boxes_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def class_prf(
    truth: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    class_id: int,
    confidence_threshold: float,
    iou_threshold: float,
    image_subset: set[str] | None = None,
) -> tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0

    image_names = sorted(image_subset or set(truth))
    for image_name in image_names:
        gt = truth[image_name]
        pred = predictions.get(
            image_name,
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "scores": np.zeros((0,), dtype=np.float32),
            },
        )

        gt_boxes = gt["boxes"][gt["labels"] == class_id]
        keep = (
            (pred["labels"] == class_id)
            & (pred["scores"] >= confidence_threshold)
        )
        pred_boxes = pred["boxes"][keep]
        pred_scores = pred["scores"][keep]

        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]

        matched: set[int] = set()
        for box in pred_boxes:
            ious = box_iou_numpy(box, gt_boxes)
            if len(ious) == 0:
                fp += 1
                continue

            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and best not in matched:
                tp += 1
                matched.add(best)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1, tp, fp, fn


def interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    values = []
    for recall_point in recall_points:
        eligible = precisions[recalls >= recall_point]
        values.append(float(np.max(eligible)) if eligible.size else 0.0)
    return float(np.mean(values))


def class_ap(
    truth: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    class_id: int,
    iou_threshold: float,
    image_subset: set[str] | None = None,
) -> float:
    image_names = sorted(image_subset or set(truth))

    gt_by_image: dict[str, np.ndarray] = {}
    total_gt = 0
    for image_name in image_names:
        gt = truth[image_name]
        boxes = gt["boxes"][gt["labels"] == class_id]
        gt_by_image[image_name] = boxes
        total_gt += len(boxes)

    if total_gt == 0:
        return float("nan")

    prediction_rows: list[tuple[float, str, np.ndarray]] = []
    for image_name in image_names:
        pred = predictions.get(
            image_name,
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "scores": np.zeros((0,), dtype=np.float32),
            },
        )
        keep = pred["labels"] == class_id
        for score, box in zip(pred["scores"][keep], pred["boxes"][keep]):
            prediction_rows.append((float(score), image_name, box))

    prediction_rows.sort(key=lambda row: row[0], reverse=True)
    matched = {
        image_name: np.zeros(len(boxes), dtype=bool)
        for image_name, boxes in gt_by_image.items()
    }

    tp = np.zeros(len(prediction_rows), dtype=np.float32)
    fp = np.zeros(len(prediction_rows), dtype=np.float32)

    for index, (_, image_name, box) in enumerate(prediction_rows):
        gt_boxes = gt_by_image[image_name]
        ious = box_iou_numpy(box, gt_boxes)

        if len(ious) == 0:
            fp[index] = 1
            continue

        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold and not matched[image_name][best]:
            tp[index] = 1
            matched[image_name][best] = True
        else:
            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recalls = cumulative_tp / max(total_gt, 1)
    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12
    )
    return interpolated_ap(recalls, precisions)


def evaluate_predictions(
    truth: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    confidence_threshold: float,
    group_name: str,
    image_subset: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        precision, recall, f1, tp, fp, fn = class_prf(
            truth,
            predictions,
            class_id,
            confidence_threshold,
            iou_threshold=0.5,
            image_subset=image_subset,
        )

        ap50 = class_ap(
            truth,
            predictions,
            class_id,
            iou_threshold=0.5,
            image_subset=image_subset,
        )
        aps = [
            class_ap(
                truth,
                predictions,
                class_id,
                iou_threshold=float(threshold),
                image_subset=image_subset,
            )
            for threshold in np.arange(0.50, 0.96, 0.05)
        ]
        valid_aps = [value for value in aps if not math.isnan(value)]
        ap5095 = float(np.mean(valid_aps)) if valid_aps else float("nan")

        row = {
            "group": group_name,
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": ap50,
            "ap50_95": ap5095,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        class_rows.append(row)
        rows.append(row)

    macro_fields = ["precision", "recall", "f1", "ap50", "ap50_95"]
    macro = {
        "group": group_name,
        "class": "macro",
        "tp": sum(int(row["tp"]) for row in class_rows),
        "fp": sum(int(row["fp"]) for row in class_rows),
        "fn": sum(int(row["fn"]) for row in class_rows),
    }
    for field in macro_fields:
        values = [
            float(row[field])
            for row in class_rows
            if not math.isnan(float(row[field]))
        ]
        macro[field] = float(np.mean(values)) if values else float("nan")
    rows.append(macro)

    return rows


def evaluate_by_manifest_group(
    dataset_root: Path,
    truth: dict[str, dict[str, np.ndarray]],
    predictions: dict[str, dict[str, np.ndarray]],
    confidence_threshold: float,
    group_by: str,
) -> list[dict[str, Any]]:
    manifest = load_manifest(dataset_root)

    if group_by == "none":
        return evaluate_predictions(
            truth,
            predictions,
            confidence_threshold,
            group_name="all_test",
        )

    groups: dict[str, set[str]] = defaultdict(set)
    for image_name in truth:
        row = manifest[image_name]
        groups[row[group_by]].add(image_name)

    rows = evaluate_predictions(
        truth,
        predictions,
        confidence_threshold,
        group_name="all_test",
    )
    for group, image_names in sorted(groups.items()):
        rows.extend(
            evaluate_predictions(
                truth,
                predictions,
                confidence_threshold,
                group_name=group,
                image_subset=image_names,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Ultralytics YOLO and RT-DETR
# ---------------------------------------------------------------------------

def train_ultralytics_model(
    model_type: str,
    weights: str,
    dataset_root: Path,
    run_dir: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Any, Path]:
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required. Install with: pip install ultralytics"
        ) from exc

    if model_type == "rtdetr":
        model = RTDETR(weights)
        learning_rate = args.rtdetr_lr
        batch = args.rtdetr_batch
        mosaic = 0.0
    else:
        model = YOLO(weights)
        learning_rate = args.yolo_lr
        batch = args.yolo_batch
        mosaic = args.yolo_mosaic

    project = run_dir.parent
    name = run_dir.name

    train_kwargs = dict(
        data=str(dataset_root / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        seed=seed,
        deterministic=True,
        optimizer=args.ultralytics_optimizer,
        lr0=learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        scale=args.scale,
        mosaic=mosaic,
        copy_paste=0.0,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    # Current Ultralytics YOLO releases can optionally apply inverse-frequency
    # class weighting through cls_pw. Keep it disabled for the baseline and run
    # a separate imbalance experiment with --cls-pw, e.g. 0.25.
    if model_type in {"yolo", "yolo_light"} and args.cls_pw > 0:
        train_kwargs["cls_pw"] = args.cls_pw

    model.train(**train_kwargs)

    best_path = project / name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Ultralytics best checkpoint not found: {best_path}")

    if model_type == "rtdetr":
        trained = RTDETR(str(best_path))
    else:
        trained = YOLO(str(best_path))
    return trained, best_path


def predict_ultralytics(
    model: Any,
    image_paths: Sequence[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    predictions = {}
    elapsed_total = 0.0

    for image_path in tqdm(image_paths, desc="Ultralytics inference"):
        start = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.ap_min_confidence,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )[0]
        elapsed_total += time.perf_counter() - start

        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

        predictions[image_path.name] = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

    milliseconds_per_tile = (
        1000.0 * elapsed_total / len(image_paths)
        if image_paths else float("nan")
    )
    return predictions, milliseconds_per_tile


def train_yolo_segmentation(
    dataset_root: Path,
    run_dir: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Any, Path, list[dict[str, Any]], float]:
    """
    Train and evaluate a YOLO26 instance-segmentation baseline.

    The prepared dataset must use YOLO segmentation labels:
        class_id x1 y1 x2 y2 ... xn yn

    The official pretrained checkpoint `yolo26n-seg.pt` is downloaded
    automatically by Ultralytics when it is not already present locally.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required. Install it with: pip install ultralytics"
        ) from exc

    seed_everything(seed)
    project = run_dir.parent
    name = run_dir.name

    model = YOLO(args.seg_weights)
    model.train(
        data=str(dataset_root / "data.yaml"),
        epochs=args.seg_epochs,
        imgsz=args.imgsz,
        batch=args.seg_batch,
        device=args.device,
        workers=args.workers,
        seed=seed,
        deterministic=True,
        optimizer=args.ultralytics_optimizer,
        lr0=args.seg_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        scale=args.scale,
        mosaic=args.seg_mosaic,
        copy_paste=args.seg_copy_paste,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    best_path = project / name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"YOLO segmentation checkpoint was not found: {best_path}"
        )

    trained_model = YOLO(str(best_path))
    metrics = trained_model.val(
        data=str(dataset_root / "data.yaml"),
        split="test",
        imgsz=args.imgsz,
        batch=args.seg_batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(project),
        name=f"{name}_test",
        exist_ok=True,
        verbose=True,
    )

    mask_metric = metrics.seg
    ap_class_indices = [
        int(class_id) for class_id in mask_metric.ap_class_index
    ]
    metric_position = {
        class_id: index
        for index, class_id in enumerate(ap_class_indices)
    }

    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        position = metric_position.get(class_id)

        if position is None:
            precision = recall = f1 = ap50 = ap50_95 = float("nan")
        else:
            precision = float(mask_metric.p[position])
            recall = float(mask_metric.r[position])
            f1 = float(mask_metric.f1[position])
            ap50 = float(mask_metric.ap50[position])
            ap50_95 = float(mask_metric.ap[position])

        rows.append(
            {
                "group": "all_test",
                "class": class_name,
                "metric_type": "mask",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50": ap50,
                "ap50_95": ap50_95,
                "tp": "",
                "fp": "",
                "fn": "",
            }
        )

    macro_row: dict[str, Any] = {
        "group": "all_test",
        "class": "macro",
        "metric_type": "mask",
        "tp": "",
        "fp": "",
        "fn": "",
    }
    for field in ("precision", "recall", "f1", "ap50", "ap50_95"):
        values = [
            float(row[field])
            for row in rows
            if not math.isnan(float(row[field]))
        ]
        macro_row[field] = float(np.mean(values)) if values else float("nan")
    rows.append(macro_row)

    speed = metrics.speed or {}
    speed_rows = [
        {
            "stage": stage,
            "ms_per_tile": float(speed.get(stage, 0.0)),
        }
        for stage in ("preprocess", "inference", "postprocess")
    ]
    milliseconds_per_tile = sum(
        float(row["ms_per_tile"]) for row in speed_rows
    )
    speed_rows.append(
        {
            "stage": "total",
            "ms_per_tile": milliseconds_per_tile,
        }
    )

    write_rows_csv(run_dir / "segmentation_test_metrics.csv", rows)
    write_rows_csv(run_dir / "segmentation_speed.csv", speed_rows)

    return trained_model, best_path, rows, milliseconds_per_tile


# ---------------------------------------------------------------------------
# Torchvision Faster R-CNN
# ---------------------------------------------------------------------------

class YoloTileDataset:
    """Torch Dataset wrapper for the prepared tiled YOLO dataset."""

    def __init__(self, dataset_root: Path, split: str):
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        self.torch = torch
        self.pil_to_tensor = pil_to_tensor
        self.dataset_root = dataset_root
        self.split = split
        self.image_paths = split_image_paths(dataset_root, split)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = open_rgb_image(image_path)
        width, height = image.size

        label_path = (
            self.dataset_root
            / "labels"
            / self.split
            / f"{image_path.stem}.txt"
        )
        boxes, labels = read_yolo_label(label_path, width, height)

        image_tensor = self.pil_to_tensor(image).float() / 255.0
        target = {
            "boxes": self.torch.as_tensor(boxes, dtype=self.torch.float32),
            # Torchvision reserves 0 for the background class.
            "labels": self.torch.as_tensor(labels + 1, dtype=self.torch.int64),
            "image_id": self.torch.tensor([index]),
            "area": self.torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                dtype=self.torch.float32,
            ),
            "iscrowd": self.torch.zeros(
                (len(boxes),), dtype=self.torch.int64
            ),
        }
        return image_tensor, target, image_path.name


def detection_collate(batch):
    images, targets, names = zip(*batch)
    return list(images), list(targets), list(names)


def create_fasterrcnn_model() -> Any:
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        input_features,
        num_classes=3,  # background + broadleaf + narrowleaf
    )
    model.transform.min_size = (768,)
    model.transform.max_size = 768
    return model


def train_fasterrcnn(
    dataset_root: Path,
    run_dir: Path,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Any, Path]:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(seed)
    device = resolve_torch_device(args.device)

    train_dataset = YoloTileDataset(dataset_root, "train")
    val_dataset = YoloTileDataset(dataset_root, "val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.faster_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_fasterrcnn_model().to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.faster_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.faster_lr_step,
        gamma=0.1,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_map = -1.0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    for epoch in range(1, args.faster_epochs + 1):
        model.train()
        epoch_losses = []

        for images, targets, _ in tqdm(
            train_loader,
            desc=f"Faster R-CNN epoch {epoch}/{args.faster_epochs}",
        ):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                losses = model(images, targets)
                loss = sum(loss_value for loss_value in losses.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch}: training loss={mean_loss:.6f}")

        if epoch % args.faster_eval_every == 0 or epoch == args.faster_epochs:
            val_predictions, _ = predict_fasterrcnn(
                model,
                val_dataset,
                device,
                args.ap_min_confidence,
            )
            val_truth = ground_truth_for_split(dataset_root, "val")
            val_rows = evaluate_predictions(
                val_truth,
                val_predictions,
                confidence_threshold=args.metric_confidence,
                group_name="validation",
            )
            val_macro = next(
                row for row in val_rows if row["class"] == "macro"
            )
            val_map = float(val_macro["ap50_95"])
            print(f"Epoch {epoch}: val macro AP@50:95={val_map:.6f}")

            if val_map > best_map:
                best_map = val_map
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_map50_95": val_map,
                        "seed": seed,
                    },
                    best_path,
                )

    if not best_path.exists():
        raise RuntimeError("No Faster R-CNN checkpoint was saved.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, best_path


def predict_fasterrcnn(
    model: Any,
    dataset: YoloTileDataset,
    device: Any,
    minimum_confidence: float,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )

    predictions = {}
    elapsed_total = 0.0
    model.eval()

    with torch.no_grad():
        for images, _, names in tqdm(loader, desc="Faster R-CNN inference"):
            images = [image.to(device) for image in images]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_total += time.perf_counter() - start

            for name, output in zip(names, outputs):
                scores = output["scores"].detach().cpu().numpy()
                keep = scores >= minimum_confidence
                predictions[name] = {
                    "boxes": output["boxes"].detach().cpu().numpy()[keep].astype(np.float32),
                    # Convert 1/2 Torchvision labels back to 0/1.
                    "labels": (
                        output["labels"].detach().cpu().numpy()[keep].astype(np.int64) - 1
                    ),
                    "scores": scores[keep].astype(np.float32),
                }

    milliseconds_per_tile = (
        1000.0 * elapsed_total / len(dataset)
        if len(dataset) else float("nan")
    )
    return predictions, milliseconds_per_tile


# ---------------------------------------------------------------------------
# Experiment orchestration and result aggregation
# ---------------------------------------------------------------------------

def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_run_metadata(
    rows: list[dict[str, Any]],
    model_name: str,
    seed: int,
    milliseconds_per_tile: float,
    checkpoint: Path,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        enriched = {
            "model": model_name,
            "seed": seed,
            **row,
            "ms_per_tile": milliseconds_per_tile,
            "checkpoint": str(checkpoint),
        }
        result.append(enriched)
    return result


def aggregate_results(
    rows: Sequence[dict[str, Any]],
    group_field: str,
) -> list[dict[str, Any]]:
    """
    Calculate mean ± standard deviation across repeated seeds.

    Rows remain separated by model, group, and class.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row[group_field]),
            str(row["class"]),
        )
        grouped[key].append(row)

    metric_fields = [
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap50_95",
        "ms_per_tile",
    ]

    summary = []
    for (model, group, class_name), items in sorted(grouped.items()):
        result = {
            "model": model,
            group_field: group,
            "class": class_name,
            "runs": len(items),
        }
        for field in metric_fields:
            values = [
                float(item[field])
                for item in items
                if not math.isnan(float(item[field]))
            ]
            result[f"{field}_mean"] = (
                statistics.mean(values) if values else float("nan")
            )
            result[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(result)
    return summary


def aggregate_across_groups_and_seeds(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Macro-average equally across held-out groups and repeated seeds.

    The aggregate excludes the synthetic 'all_test' row so a very large flight
    cannot dominate a smaller flight merely because it contains more tiles.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["group"]) == "all_test":
            continue
        grouped[(str(row["model"]), str(row["class"]))].append(row)

    metric_fields = [
        "precision",
        "recall",
        "f1",
        "ap50",
        "ap50_95",
        "ms_per_tile",
    ]
    output = []
    for (model, class_name), items in sorted(grouped.items()):
        result = {
            "model": model,
            "class": class_name,
            "group_seed_rows": len(items),
        }
        for field in metric_fields:
            values = [
                float(item[field])
                for item in items
                if not math.isnan(float(item[field]))
            ]
            result[f"{field}_macro_mean"] = (
                statistics.mean(values) if values else float("nan")
            )
            result[f"{field}_macro_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def infer_prepared_task(dataset_root: Path) -> str:
    """Read the prepared label type from settings or infer it from one label."""
    settings_path = dataset_root / "preparation_settings.json"
    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as f:
            settings = json.load(f)
        task = str(settings.get("task", "")).strip().lower()
        if task in {"detection", "segmentation"}:
            return task

    for split in ("train", "val", "test"):
        label_dir = dataset_root / "labels" / split
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            value_count = len(content.splitlines()[0].split())
            if value_count == 5:
                return "detection"
            if value_count >= 7 and value_count % 2 == 1:
                return "segmentation"
            raise ValueError(
                f"Could not identify label format in {label_path}. "
                f"The first row has {value_count} values."
            )

    raise ValueError(
        "The prepared task could not be identified because no non-empty label "
        "file or valid preparation_settings.json was found."
    )


def train_models(args: argparse.Namespace) -> None:
    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    detection_models = {"yolo", "yolo_light", "fasterrcnn", "rtdetr"}
    segmentation_models = {"yolo_seg"}
    requested = set(args.models)

    contains_detection = bool(requested & detection_models)
    contains_segmentation = bool(requested & segmentation_models)

    if contains_detection and contains_segmentation:
        raise ValueError(
            "Detection and segmentation models require different label formats. "
            "Prepare two datasets and run `yolo_seg` separately."
        )

    prepared_task = infer_prepared_task(dataset_root)
    expected_task = "segmentation" if contains_segmentation else "detection"
    if prepared_task != expected_task:
        raise ValueError(
            f"The selected models require a {expected_task} dataset, but "
            f"'{dataset_root}' contains {prepared_task} labels."
        )

    all_rows: list[dict[str, Any]] = []

    # YOLO instance-segmentation is evaluated through Ultralytics mask metrics.
    if contains_segmentation:
        for seed in args.seeds:
            seed_everything(seed)
            run_dir = output_root / f"yolo_seg_seed{seed}"

            print(f"\n{'=' * 90}")
            print(f"Training yolo_seg, seed={seed}")
            print(f"{'=' * 90}")

            _, checkpoint, rows, speed = train_yolo_segmentation(
                dataset_root=dataset_root,
                run_dir=run_dir,
                seed=seed,
                args=args,
            )
            rows = append_run_metadata(
                rows=rows,
                model_name="yolo_seg",
                seed=seed,
                milliseconds_per_tile=speed,
                checkpoint=checkpoint,
            )
            all_rows.extend(rows)
            write_rows_csv(run_dir / "test_metrics.csv", rows)

        write_rows_csv(output_root / "all_test_metrics.csv", all_rows)
        summary = aggregate_results(all_rows, "group")
        write_rows_csv(output_root / "mean_std_across_seeds.csv", summary)
        print(f"\nAll segmentation results saved to: {output_root}")
        return

    test_paths = split_image_paths(dataset_root, "test")
    test_truth = ground_truth_for_split(dataset_root, "test")

    for seed in args.seeds:
        seed_everything(seed)

        for model_name in args.models:
            run_dir = output_root / f"{model_name}_seed{seed}"
            print(f"\n{'=' * 90}")
            print(f"Training {model_name}, seed={seed}")
            print(f"{'=' * 90}")

            if model_name == "yolo":
                model, checkpoint = train_ultralytics_model(
                    model_type="yolo",
                    weights=args.yolo_weights,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    seed=seed,
                    args=args,
                )
                predictions, speed = predict_ultralytics(
                    model, test_paths, args
                )

            elif model_name == "yolo_light":
                model, checkpoint = train_ultralytics_model(
                    model_type="yolo_light",
                    weights=args.light_weights,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    seed=seed,
                    args=args,
                )
                predictions, speed = predict_ultralytics(
                    model, test_paths, args
                )

            elif model_name == "rtdetr":
                model, checkpoint = train_ultralytics_model(
                    model_type="rtdetr",
                    weights=args.rtdetr_weights,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    seed=seed,
                    args=args,
                )
                predictions, speed = predict_ultralytics(
                    model, test_paths, args
                )

            elif model_name == "fasterrcnn":
                model, checkpoint = train_fasterrcnn(
                    dataset_root,
                    run_dir,
                    seed,
                    args,
                )
                device = resolve_torch_device(args.device)
                test_dataset = YoloTileDataset(dataset_root, "test")
                predictions, speed = predict_fasterrcnn(
                    model,
                    test_dataset,
                    device,
                    args.ap_min_confidence,
                )

            else:
                raise ValueError(f"Unsupported model: {model_name}")

            rows = evaluate_by_manifest_group(
                dataset_root,
                test_truth,
                predictions,
                args.metric_confidence,
                args.group_metrics,
            )
            rows = append_run_metadata(
                rows,
                model_name,
                seed,
                speed,
                checkpoint,
            )
            all_rows.extend(rows)
            write_rows_csv(run_dir / "test_metrics.csv", rows)

    write_rows_csv(output_root / "all_test_metrics.csv", all_rows)

    summary = aggregate_results(all_rows, "group")
    write_rows_csv(output_root / "mean_std_across_seeds.csv", summary)

    group_macro = aggregate_across_groups_and_seeds(all_rows)
    write_rows_csv(
        output_root / "macro_mean_std_across_groups_and_seeds.csv",
        group_macro,
    )

    print(f"\nAll results saved to: {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help=(
            "Optional case-insensitive path fragments. Only matching acquisition "
            "folders are loaded. Omit to load all matching folders."
        ),
    )
    parser.add_argument(
        "--modality",
        choices=["rgb", "ms", "all"],
        default="rgb",
    )
    parser.add_argument(
        "--task",
        choices=["detection", "segmentation"],
        default="detection",
        help=(
            "Prepare YOLO bounding-box labels for detection or polygon labels "
            "for instance segmentation."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "folder", "flight", "field", "season", "manual"],
        default="manual",
    )
    parser.add_argument("--manual-split", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-ratio",
        type=parse_negative_ratio,
        default=0.15,
        help=(
            "Training negatives per acquisition relative to positive training "
            "source images. Example: 0.15. Use 'all' for every available "
            "training negative. Validation/test negatives are always retained."
        ),
    )
    parser.add_argument(
        "--negative-tiles-per-image",
        type=int,
        default=1,
        help=(
            "Number of random background tiles retained from each selected "
            "negative TRAINING source image. Val/test use all tiles."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--min-visibility", type=float, default=0.30)
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument("--ignore-unmapped", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolo", "yolo_light", "fasterrcnn", "rtdetr", "yolo_seg"],
        default=["yolo", "yolo_light", "fasterrcnn", "rtdetr"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument(
        "--group-metrics",
        choices=["none", "flight", "field", "season"],
        default="flight",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.20)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--metric-confidence", type=float, default=0.25)
    parser.add_argument(
        "--ap-min-confidence",
        type=float,
        default=0.001,
        help="Low threshold retained for constructing precision-recall curves.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70)

    # Exact weight names are configurable so the experiment can pin a model
    # release in the paper and requirements/environment file.
    parser.add_argument("--yolo-weights", default="yolo26m.pt")
    parser.add_argument("--light-weights", default="yolo26n.pt")
    parser.add_argument("--rtdetr-weights", default="rtdetr-l.pt")
    parser.add_argument(
        "--seg-weights",
        default="yolo26n-seg.pt",
        help=(
            "Pretrained YOLO instance-segmentation checkpoint. Ultralytics "
            "downloads official weights automatically when needed."
        ),
    )

    parser.add_argument("--ultralytics-optimizer", default="AdamW")
    parser.add_argument("--yolo-batch", type=int, default=4)
    parser.add_argument("--yolo-lr", type=float, default=0.001)
    parser.add_argument("--yolo-mosaic", type=float, default=0.0)
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=0.0,
        help="Optional YOLO class-weighting power; 0 disables it.",
    )
    parser.add_argument("--rtdetr-batch", type=int, default=4)
    parser.add_argument("--rtdetr-lr", type=float, default=0.003)

    parser.add_argument("--seg-epochs", type=int, default=200)
    parser.add_argument("--seg-batch", type=int, default=4)
    parser.add_argument("--seg-lr", type=float, default=0.001)
    parser.add_argument("--seg-mosaic", type=float, default=0.0)
    parser.add_argument("--seg-copy-paste", type=float, default=0.0)

    parser.add_argument("--faster-epochs", type=int, default=50)
    parser.add_argument("--faster-batch", type=int, default=2)
    parser.add_argument("--faster-lr", type=float, default=0.005)
    parser.add_argument("--faster-lr-step", type=int, default=20)
    parser.add_argument("--faster-eval-every", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CamelinaWeed leakage-safe multi-detector experiment pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-folders")
    list_parser.add_argument("--input", type=Path, required=True)
    list_parser.add_argument(
        "--modality",
        choices=["rgb", "ms", "all"],
        default="all",
    )

    prepare_parser = subparsers.add_parser("prepare")
    add_prepare_arguments(prepare_parser)

    train_parser = subparsers.add_parser("train")
    add_train_arguments(train_parser)

    all_parser = subparsers.add_parser("all")
    add_prepare_arguments(all_parser)
    # Avoid duplicate --output: use --results-output for training in all mode.
    all_parser.add_argument("--results-output", type=Path, required=True)
    all_parser.add_argument(
        "--models",
        nargs="+",
        choices=["yolo", "yolo_light", "fasterrcnn", "rtdetr", "yolo_seg"],
        default=["yolo", "yolo_light", "fasterrcnn", "rtdetr"],
    )
    all_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    all_parser.add_argument(
        "--group-metrics",
        choices=["none", "flight", "field", "season"],
        default="flight",
    )
    all_parser.add_argument("--device", default="0")
    all_parser.add_argument("--workers", type=int, default=4)
    all_parser.add_argument("--imgsz", type=int, default=768)
    all_parser.add_argument("--epochs", type=int, default=200)
    all_parser.add_argument("--patience", type=int, default=60)
    all_parser.add_argument("--scale", type=float, default=0.20)
    all_parser.add_argument("--weight-decay", type=float, default=0.0005)
    all_parser.add_argument("--metric-confidence", type=float, default=0.25)
    all_parser.add_argument("--ap-min-confidence", type=float, default=0.001)
    all_parser.add_argument("--nms-iou", type=float, default=0.70)
    all_parser.add_argument("--yolo-weights", default="yolo26m.pt")
    all_parser.add_argument("--light-weights", default="yolo26n.pt")
    all_parser.add_argument("--rtdetr-weights", default="rtdetr-l.pt")
    all_parser.add_argument("--seg-weights", default="yolo26n-seg.pt")
    all_parser.add_argument("--ultralytics-optimizer", default="AdamW")
    all_parser.add_argument("--yolo-batch", type=int, default=4)
    all_parser.add_argument("--yolo-lr", type=float, default=0.001)
    all_parser.add_argument("--yolo-mosaic", type=float, default=0.0)
    all_parser.add_argument("--cls-pw", type=float, default=0.0)
    all_parser.add_argument("--rtdetr-batch", type=int, default=4)
    all_parser.add_argument("--rtdetr-lr", type=float, default=0.003)
    all_parser.add_argument("--seg-epochs", type=int, default=200)
    all_parser.add_argument("--seg-batch", type=int, default=4)
    all_parser.add_argument("--seg-lr", type=float, default=0.001)
    all_parser.add_argument("--seg-mosaic", type=float, default=0.0)
    all_parser.add_argument("--seg-copy-paste", type=float, default=0.0)
    all_parser.add_argument("--faster-epochs", type=int, default=50)
    all_parser.add_argument("--faster-batch", type=int, default=2)
    all_parser.add_argument("--faster-lr", type=float, default=0.005)
    all_parser.add_argument("--faster-lr-step", type=int, default=20)
    all_parser.add_argument("--faster-eval-every", type=int, default=5)

    return parser


def list_folders(args: argparse.Namespace) -> None:
    root = args.input.resolve()
    acquisitions = discover_acquisitions(root, [], args.modality)
    print("Available acquisition folders:")
    for acquisition in acquisitions:
        print(acquisition.relative_to(root).as_posix())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list-folders":
        list_folders(args)
    elif args.command == "prepare":
        prepare_dataset(args)
    elif args.command == "train":
        train_models(args)
    elif args.command == "all":
        prepared_output = args.output
        results_output = args.results_output
        prepare_dataset(args)

        # Adapt all-mode namespace to the train function.
        args.dataset = prepared_output
        args.output = results_output
        train_models(args)
    else:
        parser.error("Unknown command")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)