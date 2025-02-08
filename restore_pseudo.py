import os
import math

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.model_utils import transformation, load_model, DEVICE
from utils.image_utils import create_path, rotate_with_mask, IMG_SIZE


def pseudo_label(model, image_path, transform, num_samples=10):
    """
    Pseudo labeling single image
    """
    image = Image.open(image_path).convert("RGB")
    transformed = transform(image).unsqueeze(0).to(DEVICE)
    angles = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            output = model(transformed)
            sin_pred = output[0, 0].item()
            cos_pred = output[0, 1].item()
            angle_rad = math.atan2(sin_pred, cos_pred)
            angle_deg = math.degrees(angle_rad) % 360
            angles.append(angle_deg)

    mean_angle = np.mean(angles)
    variance = np.var(angles)
    return mean_angle, variance


def restore_imgs_with_pl(
        model,
        input_dir,
        output_dir,
        num_samples=10,
        threshold=1e-10,
        output_format="PNG",
):
    """
    Restore images with pseudo label
    """
    create_path(output_dir, is_dir=True)

    total_files = 0
    _, val_transform = transformation()
    model.eval()

    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        total_files += 1
        input_path = os.path.join(input_dir, filename)
        angle, var = pseudo_label(
            model=model,
            image_path=input_path,
            transform=val_transform,
            num_samples=num_samples
        )

        if var < threshold:
            output_filename = f"{filename.split(".")[0]}.{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)

            with Image.open(input_path) as img:
                if img.size != (IMG_SIZE, IMG_SIZE):
                    print(f"Size does not match: {filename}")
                    continue

                rotated_img = rotate_with_mask(img, int(angle))
                if rotated_img:
                    rotated_img.save(output_path, output_format, quality=100)

    print("\nProcessing finished!")


if __name__ == '__main__':
    MODEL_NAME = "conv_0203_ft_0205"
    INPUT_DIR = "caps/unlabeled_caps"
    OUTPUT_DIR = "caps/tmp"
    DROPOUT_RATE = 0.5
    THRESHOLD = 1e-10

    model = load_model(
        model_path=f"models/{MODEL_NAME}.pth",
        dropout_rate=DROPOUT_RATE
    )
    restore_imgs_with_pl(
        model=model,
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
    )
