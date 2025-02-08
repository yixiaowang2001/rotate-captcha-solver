import torch

from utils.crawler_utils import start_driver, solve_captcha, batch_solve_captcha
from utils.model_utils import build_model, transformation

MODEL_NAME = "conv_0203"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 500  # Change to None for single real test
SAVE_PATH = None


_, transform = transformation()
model = build_model(0.5, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location=DEVICE, weights_only=True))
model.eval()

if N:
    driver = start_driver(headless=False)
    success_count, total_count = batch_solve_captcha(driver, N, model, transform, save_path=SAVE_PATH)
    print(f"Success times: {success_count}/{total_count}, "
          f"success rate: {success_count / total_count * 100:.2f}%")
    driver.quit()
else:
    driver = start_driver(headless=False)
    success = solve_captcha(driver, model, transform)
    print("Success" if success else "Failure")
    driver.quit()