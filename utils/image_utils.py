import os
import shutil
import random

from PIL import Image, ImageDraw
from tqdm import tqdm

IMG_SIZE = 152


class CircleMask:
    """
    Image mask
    """

    def __init__(self, img_size):
        self.mask = Image.new("L", (img_size, img_size), 0)
        draw = ImageDraw.Draw(self.mask)
        draw.ellipse((0, 0, img_size, img_size), fill=255)

    def __call__(self, img):
        img.putalpha(self.mask)
        return img.convert("RGB")


def create_path(path, is_dir=False):
    """
    Create a new directory (overwrite the previous one)
    """
    if os.path.exists(path):
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)
    if is_dir:
        os.makedirs(path)


def apply_mask(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Apply mask to an image (with white background)
    """
    circle_mask = CircleMask(target_size[0])
    white_bg = Image.new("RGB", target_size, (255, 255, 255))
    return Image.composite(image.convert("RGB"), white_bg, circle_mask.mask)


def balanced_random_rotation_fn(total_images, n_per_image, seed=None):
    """
    Randomly rotate images
    """
    if seed is not None:
        random.seed(seed)

    total_angles_needed = total_images * n_per_image
    global_angles = [random.uniform(0, 360) for _ in range(total_angles_needed)]
    random.shuffle(global_angles)

    def rotation_fn():
        if not hasattr(rotation_fn, "angle_pool"):
            rotation_fn.angle_pool = global_angles
        if rotation_fn.angle_pool:
            return rotation_fn.angle_pool.pop()
        else:
            raise RuntimeError("No more angles available in the pool!")

    return rotation_fn


def fixed_rotation_fn(interval, noise=False, seed=None):
    """
    Rotate images with fixed angle interval
    """
    if seed is not None:
        random.seed(seed)
    angles = list(range(0, 360, interval))

    def rotation_fn():
        angle = random.choice(angles)
        if noise:
            angle += random.uniform(-5, 5)
        angle %= 360
        return angle

    return rotation_fn


def generate_imgs(
        input_dir,
        image_output_dir,
        total_output_images,
        rotation_fn,
        n_per_image,
        target_size=(IMG_SIZE, IMG_SIZE)
):
    """
    Generate images
    """
    create_path(image_output_dir, is_dir=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    total_images = len(images)
    max_possible_images = total_images * n_per_image

    if max_possible_images < total_output_images:
        raise ValueError(f"Cannot generate {total_output_images} images. "
                         f"Maximum possible with current settings is {max_possible_images}.")

    output_count = 0
    with tqdm(total=total_output_images, desc="Generating images") as pbar:
        for image_name in images:
            image_path = os.path.join(input_dir, image_name)
            try:
                image = Image.open(image_path).resize(target_size, Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
                continue

            for _ in range(n_per_image):
                if output_count >= total_output_images:
                    return

                angle = rotation_fn()
                clockwise_angle = -angle

                rotated_image = image.rotate(
                    clockwise_angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor=(255, 255, 255) if image.mode == "RGB" else 255
                )

                masked_image = apply_mask(rotated_image, target_size)

                angle_str = f"{angle:.2f}".replace(".", "_")
                output_image_name = f"{os.path.splitext(image_name)[0]}_rot{angle_str}_{output_count}.jpg"
                output_image_path = os.path.join(image_output_dir, output_image_name)
                masked_image.save(output_image_path)

                output_count += 1
                pbar.update(1)


def rotate_with_mask(img, angle, size=IMG_SIZE):
    """
    Rotate an image
    """
    background = Image.new("RGB", (size, size), (255, 255, 255))
    mask = CircleMask(size)
    content = img.convert("RGBA")
    content.putalpha(mask.mask)
    rotated = content.rotate(angle, resample=Image.BICUBIC, expand=False)
    background.paste(rotated.convert("RGB"), (0, 0), rotated)
    return background
