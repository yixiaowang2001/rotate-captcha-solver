from utils.image_utils import balanced_random_rotation_fn, fixed_rotation_fn, generate_imgs


total_images = 1000
n_per_image = 1
seed = 42

rotation_fn = balanced_random_rotation_fn(total_images, n_per_image, seed=seed)
# rotation_fn = fixed_rotation_fn(interval=1, seed=seed)

generate_imgs(
    input_dir="imgs/raw_test_imgs",
    image_output_dir="imgs/tmp",
    total_output_images=total_images,
    rotation_fn=rotation_fn,
    n_per_image=n_per_image
)
