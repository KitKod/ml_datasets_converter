import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate VOC Dataset.")
    parser.add_argument("--src", required=True, help="Path to the VOC dataset.")
    return parser.parse_args()


def validate_annotation(root, class_dict, image_id):
    annotation_file = os.path.join(root, f"Annotations/{image_id}")
    print(f"Start validating annotation_file={annotation_file}")

    objects = ET.parse(annotation_file).findall("object")
    boxes = []
    labels = []
    is_difficult = []
    for object in objects:
        class_name = object.find("name").text.strip()  # .lower().strip()
        # we're only concerned with clases in our list
        if class_name in class_dict:
            bbox = object.find("bndbox")

            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1
            boxes.append([x1, y1, x2, y2])

            labels.append(class_dict[class_name])

            # retrieve <difficult> element
            is_difficult_obj = object.find("difficult")
            is_difficult_str = "0"

            if is_difficult_obj is not None:
                is_difficult_str = object.find("difficult").text

            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        else:
            print(
                f"warning - image {image_id} has object with unknown class '{class_name}'"
            )

    return (
        np.array(boxes, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(is_difficult, dtype=np.uint8),
    )


def validate_labels_file(root):
    label_file_name = os.path.join(root, "labels.txt")
    class_names = None

    if os.path.isfile(label_file_name):
        classes = []

        # classes should be a line-separated list
        with open(label_file_name, "r") as infile:
            for line in infile:
                classes.append(line.rstrip())

        # prepend BACKGROUND as first class
        classes.insert(0, "BACKGROUND")
        class_names = tuple(classes)
        print(f"VOC Labels read from file:  {class_names}")

    else:
        print("No labels file, using default VOC classes.")

    class_dict = {class_name: i for i, class_name in enumerate(class_names)}

    return class_dict


def validate_image_ids(root, main_file):
    trainval_file = os.path.join(root, f"ImageSets/Main/{main_file}")
    jpeg_images_dir = os.path.join(root, "JPEGImages")
    annotations_dir = os.path.join(root, "Annotations")

    if not os.path.isfile(trainval_file):
        print(f"trainval.txt not found at {trainval_file}.")
        return

    with open(trainval_file, "r") as file:
        img_ids = [line.strip() for line in file.readlines()]

    missing_in_images = []
    missing_in_annotations = []

    for img_id in img_ids:
        image_file = os.path.join(jpeg_images_dir, f"{img_id}.jpg")
        annotation_file = os.path.join(annotations_dir, f"{img_id}.xml")

        if not os.path.isfile(image_file):
            missing_in_images.append(img_id)

        if not os.path.isfile(annotation_file):
            missing_in_annotations.append(img_id)

    if missing_in_images:
        print(f"[{main_file}] Missing images in JPEGImages: {missing_in_images}")
    if missing_in_annotations:
        print(
            f"[{main_file}] Missing annotations in Annotations: {missing_in_annotations}"
        )

    if not missing_in_images and not missing_in_annotations:
        print(f"All img-ids in {main_file} are present in JPEGImages and Annotations.")


def main(root):
    class_dict = validate_labels_file(root)

    for image_id in os.listdir(os.path.join(root, "Annotations")):
        res = validate_annotation(root, class_dict, image_id)
        print(res)

    # trainval.txt
    validate_image_ids(root, "trainval.txt")
    validate_image_ids(root, "test.txt")


if __name__ == "__main__":
    args = parse_args()
    root = args.src
    main(root)
