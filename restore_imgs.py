import os
import random
import re
import traceback

from PIL import Image

from utils.image_utils import create_path, rotate_with_mask, IMG_SIZE


def restore_imgs(
        input_dir,
        train_dir,
        val_dir,
        test_dir,
        split_ratio,
        output_format="PNG"
):
    """
    Restore angles for labeled imgs
    """
    for output_dir in [train_dir, val_dir, test_dir]:
        create_path(output_dir, is_dir=True)

    total_files = 0
    success_count = 0
    error_files = []

    unique_img_ids = set()
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(("png", "jpg", "jpeg")):
            continue
        try:
            img_id = filename.split("_")[0]
            if img_id.isdigit():
                unique_img_ids.add(img_id)
        except:
            pass

    img_ids = list(unique_img_ids)
    random.shuffle(img_ids)
    total_ids = len(img_ids)
    train_count = int(total_ids * split_ratio[0])
    val_count = int(total_ids * split_ratio[1])

    train_ids = set(img_ids[:train_count])
    val_ids = set(img_ids[train_count:train_count + val_count])
    test_ids = set(img_ids[train_count + val_count:])

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(("png", "jpg", "jpeg")):
            continue
        total_files += 1

        try:
            img_id = filename.split("_")[0]
            if not img_id.isdigit() or img_id not in train_ids | val_ids | test_ids:
                error_files.append(filename)
                continue
        except Exception as e:
            print(f"Failed to parse filename: {filename} ({str(e)})")
            error_files.append(filename)
            continue

        angle_match = re.search(r"_rot(\d+)_", filename)
        if not angle_match:
            print(f"Failed to find rotate angle in filename: {filename}")
            error_files.append(filename)
            continue
        angle = angle_match.group(1)

        if img_id in train_ids:
            output_dir = train_dir
        elif img_id in val_ids:
            output_dir = val_dir
        else:
            output_dir = test_dir

        try:
            input_path = os.path.join(input_dir, filename)
            output_filename = f"{img_id}_rot{angle}.{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)

            with Image.open(input_path) as img:
                if img.size != (IMG_SIZE, IMG_SIZE):
                    print(f"Size does not match: {filename}")
                    error_files.append(filename)
                    continue

                rotated_img = rotate_with_mask(img, int(angle))
                rotated_img.save(output_path, output_format, quality=100)
                success_count += 1
        except Exception as e:
            print(f"Failed to process: {filename}\n{'-' * 20}\n{traceback.format_exc()}{'-' * 20}")
            error_files.append(filename)

    print("\nProcessing finished!")
    print(f"Total input files: {total_files}")
    print(f"Success files: {success_count}")
    print(f"Error files: {len(error_files)}")
    print(f"Train: {len(os.listdir(train_dir))}")
    print(f"Val: {len(os.listdir(val_dir))}")
    print(f"Test: {len(os.listdir(test_dir))}")


if __name__ == '__main__':
    INPUT_FOLDER = "caps/raw_labeled_caps"
    TRAIN_DIR = "caps/train_caps"
    VAL_DIR = "caps/val_caps"
    TEST_DIR = "caps/test_caps"
    SPLIT_RATIO = (0.8, 0.1, 0.1)

    restore_imgs(
        input_dir=INPUT_FOLDER,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        split_ratio=SPLIT_RATIO
    )
