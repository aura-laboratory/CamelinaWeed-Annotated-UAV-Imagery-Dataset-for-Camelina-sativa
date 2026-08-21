#!/usr/bin/env python3

"""
Visualize COCO polygon annotations on UAV/drone images.

Supports:
- Single image visualization
- Batch visualization for all images
- Optional category labels
- Save annotated images
- Interactive display with OpenCV

Example usage:

1. Single image
python visualize_annotations.py \
--input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
--image "DJI_0051_JPG.rf.nh87J09No53uRJWaoKVC.JPG" \
--show-labels \
--display

2. Single image + save
python visualize_annotations.py \
--input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
--image "DJI_0051_JPG.rf.nh87J09No53uRJWaoKVC.JPG" \
--show-labels \
--save

3. All images
python visualize_annotations.py \
--input "Summer 2025/Thessaloniki/Phantom Flight at 10 m Altitude/Annotated" \
--all \
--show-labels \
--save
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def load_coco_annotations(annotation_path):
    """
    Load COCO annotation file.

    Returns:
        images:
            Dictionary mapping image_id -> image metadata

        categories:
            Dictionary mapping category_id -> category name

        annotations_by_image:
            Dictionary mapping image_id -> list of annotations
    """

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map image id -> image metadata
    images = {
        img["id"]: img
        for img in coco["images"]
    }

    # Map category id -> category name
    categories = {
        cat["id"]: cat["name"]
        for cat in coco["categories"]
    }

    # Group annotations by image id
    annotations_by_image = {}

    for ann in coco["annotations"]:

        image_id = ann["image_id"]

        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []

        annotations_by_image[image_id].append(ann)

    return images, categories, annotations_by_image


def generate_colors(category_ids):
    """
    Generate consistent random colors for each category.

    Using a fixed seed ensures that categories
    always get the same color across runs.
    """

    random.seed(42)

    colors = {}

    for category_id in category_ids:

        colors[category_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

    return colors


def find_annotation_file(folder_path):
    """
    Automatically locate COCO annotation JSON file.

    Looks for common annotation filenames first,
    otherwise returns the first JSON file found.
    """

    candidates = [
        "_annotations.coco.json",
        "annotations.json",
        "_annotations.json",
    ]

    # Check common filenames
    for candidate in candidates:

        path = folder_path / candidate

        if path.exists():
            return path

    # Fallback: first JSON file in folder
    json_files = list(folder_path.glob("*.json"))

    if json_files:
        return json_files[0]

    raise FileNotFoundError(
        "No annotation JSON file found."
    )


def find_image(folder_path, image_name):
    """
    Find image either:
    - in dataset root
    - inside an 'images' subfolder
    """

    possible_paths = [
        folder_path / image_name,
        folder_path / "images" / image_name,
    ]

    for path in possible_paths:

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Image not found: {image_name}"
    )


def draw_annotations(
    image,
    annotations,
    categories,
    colors,
    show_labels=False,
    alpha=0.35,
):
    """
    Draw polygon annotations on image.

    Steps:
    1. Draw filled polygons on overlay
    2. Blend overlay with original image
    3. Draw polygon borders
    4. Optionally draw labels
    """

    overlay = image.copy()

    # ---------------------------------------------------
    # Draw filled polygons on overlay image
    # ---------------------------------------------------
    for ann in annotations:

        category_id = ann["category_id"]

        category_name = categories.get(
            category_id,
            "unknown",
        )

        color = colors[category_id]

        segmentations = ann.get(
            "segmentation",
            [],
        )

        # Skip unsupported formats
        if not isinstance(segmentations, list):
            continue

        for segmentation in segmentations:

            # Need at least 3 points
            if len(segmentation) < 6:
                continue

            polygon = np.array(segmentation)
            polygon = polygon.reshape(-1, 2)
            polygon = polygon.astype(np.int32)

            # Draw filled polygon
            cv2.fillPoly(
                overlay,
                [polygon],
                color,
            )

    # ---------------------------------------------------
    # Blend overlay with original image
    # ---------------------------------------------------
    result = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0,
    )

    # ---------------------------------------------------
    # Draw borders and labels
    # ---------------------------------------------------
    for ann in annotations:

        category_id = ann["category_id"]

        category_name = categories.get(
            category_id,
            "unknown",
        )

        color = colors[category_id]

        segmentations = ann.get(
            "segmentation",
            [],
        )

        if not isinstance(segmentations, list):
            continue

        for segmentation in segmentations:

            if len(segmentation) < 6:
                continue

            polygon = np.array(segmentation)
            polygon = polygon.reshape(-1, 2)
            polygon = polygon.astype(np.int32)

            # Polygon border
            cv2.polylines(
                result,
                [polygon],
                isClosed=True,
                color=color,
                thickness=2,
            )

            # Optional labels
            if show_labels:

                x, y = polygon[0]

                text = category_name

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.25
                thickness = 4

                # White outline for visibility
                cv2.putText(
                    result,
                    text,
                    (x, y - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness + 3,
                    cv2.LINE_AA,
                )

                # Main black text
                cv2.putText(
                    result,
                    text,
                    (x, y - 10),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

    return result


def resize_for_display(image, max_width=1600):
    """
    Resize large UAV images for display.

    Prevents oversized OpenCV windows.
    """

    h, w = image.shape[:2]

    if w <= max_width:
        return image

    scale = max_width / w

    new_size = (
        int(w * scale),
        int(h * scale),
    )

    return cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA,
    )


def process_image(
    image_info,
    folder_path,
    annotations,
    categories,
    colors,
    save=False,
    output_dir=None,
    show_labels=False,
    display=False,
):
    """
    Process a single image:
    - load image
    - draw annotations
    - optionally save
    - optionally display
    """

    image_path = find_image(
        folder_path,
        image_info["file_name"],
    )

    image = cv2.imread(str(image_path))

    if image is None:

        print(
            f"Could not read image: {image_path}"
        )

        return

    annotated = draw_annotations(
        image=image,
        annotations=annotations,
        categories=categories,
        colors=colors,
        show_labels=show_labels,
    )

    # ---------------------------------------------------
    # Save output image
    # ---------------------------------------------------
    if save:

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir /
            image_info["file_name"]
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print(f"Saved: {output_path}")

    # ---------------------------------------------------
    # Display image with OpenCV
    # ---------------------------------------------------
    if display:

        display_image = resize_for_display(
            annotated
        )

        cv2.imshow(
            "Annotation Visualization",
            display_image,
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """
    Main entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Visualize COCO polygon annotations."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to annotated dataset folder.",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Specific image filename.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all images.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save visualized images.",
    )

    parser.add_argument(
        "--output",
        default="visualized_annotations",
        help="Output directory.",
    )

    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show category labels.",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display images.",
    )

    args = parser.parse_args()

    folder_path = Path(args.input)

    # Load annotation data
    annotation_file = find_annotation_file(
        folder_path
    )

    (
        images,
        categories,
        annotations_by_image,
    ) = load_coco_annotations(annotation_file)

    # Generate category colors
    colors = generate_colors(
        categories.keys()
    )

    output_dir = Path(args.output)

    # ---------------------------------------------------
    # SINGLE IMAGE MODE
    # ---------------------------------------------------
    if args.image:

        selected = [
            img
            for img in images.values()
            if img["file_name"] == args.image
        ]

        if not selected:

            raise ValueError(
                f"Image not found in annotations: "
                f"{args.image}"
            )

    # ---------------------------------------------------
    # ALL IMAGES MODE
    # ---------------------------------------------------
    elif args.all:

        selected = list(images.values())

    else:

        raise ValueError(
            "Use either --image IMAGE_NAME "
            "or --all"
        )

    # ---------------------------------------------------
    # Process selected images
    # ---------------------------------------------------
    for image_info in selected:

        image_id = image_info["id"]

        anns = annotations_by_image.get(
            image_id,
            [],
        )

        process_image(
            image_info=image_info,
            folder_path=folder_path,
            annotations=anns,
            categories=categories,
            colors=colors,
            save=args.save,
            output_dir=output_dir,
            show_labels=args.show_labels,
            display=args.display,
        )


if __name__ == "__main__":
    main()
