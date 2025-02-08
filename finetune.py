from torch.utils.data import DataLoader

from utils.model_utils import transformation, RegressionDataset, load_model, train_and_evaluate, MixedDataset

BATCH_SIZE = 32
EPOCHS = 60
FEATURE_LR = 1e-5
CLS_LR = 5e-5
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.5
FT_MODEL_NAME = "test_ft"
PRETRAINED_MODEL_NAME = "conv_0201"
INPUT_TRAIN_DIR = "caps/train_caps"
INPUT_VAL_DIR = "caps/val_caps"
# INPUT_PSEUDO_DIR = "caps/pseudo_train_caps"  # For pseudo label training


train_transform, val_transform = transformation()
train_dataset = RegressionDataset(INPUT_TRAIN_DIR, transform=train_transform)
# mixed_train_dataset = MixedDataset(INPUT_TRAIN_DIR, INPUT_PSEUDO_DIR, train_transform)  # For pseudo label training
val_dataset = RegressionDataset(INPUT_VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = load_model(
    model_path=f"models/{PRETRAINED_MODEL_NAME}.pth",
    dropout_rate=DROPOUT_RATE,
    freeze_layers=True
)
best_mae = train_and_evaluate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_name=FT_MODEL_NAME,
    epochs=EPOCHS,
    feature_lr=FEATURE_LR,
    cls_lr=CLS_LR,
    weight_decay=WEIGHT_DECAY
)
