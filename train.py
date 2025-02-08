from torch.utils.data import DataLoader

from utils.model_utils import transformation, RegressionDataset, build_model, train_and_evaluate

BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
MODEL_NAME = "test"
INPUT_TRAIN_DIR = "imgs/train"
INPUT_VAL_DIR = "imgs/val"


train_transform, val_transform = transformation()
train_dataset = RegressionDataset(INPUT_TRAIN_DIR, transform=train_transform)
val_dataset = RegressionDataset(INPUT_VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = build_model(DROPOUT_RATE)
best_mae = train_and_evaluate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    model_name=MODEL_NAME,
    epochs=EPOCHS,
    cls_lr=LEARNING_RATE
)
