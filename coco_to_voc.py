import json
import os
import argparse
import shutil
from tqdm import tqdm
from xmltodict import unparse

# BBOX_OFFSET: Switch between 0-based and 1-based bbox.
# The COCO dataset is in 0-based format, while the VOC dataset is 1-based.
# To keep 0-based, set it to 0. To convert to 1-based, set it to 1.
BBOX_OFFSET = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to VOC format.")
    parser.add_argument("--src", required=True, help="Path to the COCO dataset.")
    parser.add_argument(
        "--dst", required=True, help="Path to the output VOC format directory."
    )
    return parser.parse_args()


args = parse_args()
src_base = args.src
dst_base = os.path.join("out", "VOCdevkit", "VOC2012", args.dst)

dst_dirs = {
    x: os.path.join(dst_base, x) for x in ["Annotations", "ImageSets", "JPEGImages"]
}
dst_dirs["ImageSets"] = os.path.join(dst_dirs["ImageSets"], "Main")
for k, d in dst_dirs.items():
    os.makedirs(d, exist_ok=True)


def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOC2012",
            "segmented": "0",
            "owner": {"name": "Mykyta Kamak"},
            "source": {
                "database": "The COCO 2017 database",
                "annotation": "COCO 2017",
                "image": "unknown",
            },
            "size": {"width": width, "height": height, "depth": depth},
            "object": [],
        }
    }


def base_object(size_info, name, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info["width"]
    height = size_info["height"]

    x1 = max(x1, 0) + BBOX_OFFSET
    y1 = max(y1, 0) + BBOX_OFFSET
    x2 = min(x2, width - 1) + BBOX_OFFSET
    y2 = min(y2, height - 1) + BBOX_OFFSET

    return {
        "name": name,
        "pose": "Unspecified",
        "truncated": BBOX_OFFSET,
        "difficult": "0",
        "bndbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
    }


sets = {
    "trainval": {
        "ann_file": os.path.join(src_base, "train/_annotations.coco.json"),
        "img_dir": os.path.join(src_base, "train"),
    },
    "test": {
        "ann_file": os.path.join(src_base, "test/_annotations.coco.json"),
        "img_dir": os.path.join(src_base, "test"),
    },
}

# Load categories from the training annotations (assuming categories are the same)
cate = {
    x["id"]: x["name"]
    for x in json.load(open(sets["trainval"]["ann_file"]))["categories"]
}

# Initialize a global image ID counter and mapping
next_image_id = 1
id_mapping = {}  # Maps original image IDs to new unique IDs

for stage, filenames in sets.items():
    ann_file = filenames["ann_file"]
    img_dir = filenames["img_dir"]
    print("===> Parse", ann_file)
    data = json.load(open(ann_file))

    images = {}
    for im in tqdm(data["images"], desc=f"-> Parse Images ({stage})"):
        # Assign new unique ID
        original_id = im["id"]
        new_id = next_image_id
        next_image_id += 1
        id_mapping[original_id] = new_id

        # Update filename to match new ID
        new_filename = f"{str(new_id).zfill(12)}.jpg"
        img = base_dict(new_filename, im["width"], im["height"], 3)
        images[new_id] = img

        # Copy image to JPEGImages folder with new filename
        src_image_path = os.path.join(img_dir, im["file_name"])
        dst_image_path = os.path.join(dst_dirs["JPEGImages"], new_filename)
        if not os.path.exists(dst_image_path):
            shutil.copy2(src_image_path, dst_image_path)

    for an in tqdm(data["annotations"], desc=f"-> Parse Annotations ({stage})"):
        # Get the new image ID
        new_image_id = id_mapping[an["image_id"]]
        ann = base_object(
            images[new_image_id]["annotation"]["size"],
            cate[an["category_id"]],
            an["bbox"],
        )
        images[new_image_id]["annotation"]["object"].append(ann)

    # Write Annotations to XML files for each image
    for k, im in tqdm(images.items(), desc=f"-> Write Annotations ({stage})"):
        im["annotation"]["object"] = im["annotation"]["object"] or [None]
        unparse(
            im,
            open(
                os.path.join(dst_dirs["Annotations"], f"{str(k).zfill(12)}.xml"),
                "w",
            ),
            full_document=False,
            pretty=True,
        )

    # Generate separate txt files for trainval and test sets
    print(f"-> Write image sets for {stage}")
    with open(os.path.join(dst_dirs["ImageSets"], f"{stage}.txt"), "w") as f:
        f.writelines([f"{str(k).zfill(12)}\n" for k in images.keys()])

print("Conversion Complete")
