from torch.utils.data import DataLoader

from utils.model_utils import transformation, RegressionDataset, load_model, evaluate

BATCH_SIZE = 128
DROPOUT_RATE = 0.5
INPUT_TEST_DIRS = [
    "imgs/test",
    "caps/test_caps"
]
MODEL_NAMES = [
    "conv_0203",
    "conv_0203_ft_0205",
    "conv_0203_ft_0205_pl_0206"
]


_, val_transform = transformation()
for test_dir in INPUT_TEST_DIRS:
    print(f"\nTesting {test_dir}:")
    test_dataset = RegressionDataset(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for model_name in MODEL_NAMES:
        model = load_model(
            model_path=f"models/{model_name}.pth",
            dropout_rate=DROPOUT_RATE
        )

        test_mae, _ = evaluate(model, test_loader)
        print(f"{model_name} MAE: {test_mae:.2f}°")