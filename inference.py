import pandas as pd

from utils.model_utils import transformation, load_model, predict_angle, predict_folder


MODEL_NAME = "conv_0203_ft_0205"
DROPOUT_RATE = 0.5

_, val_transform = transformation()
model = load_model(
    model_path=f"models/{MODEL_NAME}.pth",
    dropout_rate=DROPOUT_RATE
)

# Predict single image
single_image_path = "caps/unlabeled_caps/1210.png"

angle = predict_angle(
    image_input=single_image_path,
    model=model,
    transform=val_transform,
    input_type="path"
)
print(f"Predicted angle: {angle:.2f}°. Image should be rotated: {360 - angle:.2f}°")

# # Predict images for the entire folder
# folder_path = "caps/test_caps"
# batch_size = 64
#
# results = predict_folder(
#     model=model,
#     folder_path=folder_path,
#     transform=val_transform,
#     batch_size=batch_size
# )
# for filename, angle in results.items():
#     print(f"{filename}: {angle:.2f}°")
#
# # Save results to csv
# df = pd.DataFrame(list(results.items()), columns=["Filename", "Angle"])
# df.to_csv("predictions.csv", index=False)
