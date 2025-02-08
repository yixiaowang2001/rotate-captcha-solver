import os
import random
import time

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from utils.model_utils import predict_angle

TEST_PAGE_LINK = "https://wappass.baidu.com/static/captcha/tuxing.html?ak=2ef521ec36290baed33d66de9b16f625&backurl=http%3A%2F%2Ftieba.baidu.com%2Ff%3Fkw%3D%25E5%25AD%2599%25E7%25AC%2591%25E5%25B7%259D&timestamp=1736639935&signature=e290a3fb82d7621858aab289cb845d7f"


def start_driver(headless=True):
    """
    Initialize webdriver
    """
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--disable-dev-shm-usage")

    return webdriver.Chrome(options=chrome_options)


def find_cap(driver, max_attempts=10):
    """
    Find captcha image
    """
    driver.get(TEST_PAGE_LINK)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "passMod_verify-container"))
        )
    except Exception as e:
        print(f"Error: Failed to load page - {e}")
        return None

    for _ in range(max_attempts):
        try:
            captcha_image = driver.find_element(By.CLASS_NAME, "passMod_spin-background")
            z_index = driver.find_element(By.CLASS_NAME, "passMod_verify-container").value_of_css_property("z-index")
            image_src = captcha_image.get_attribute("src")

            if z_index == "9" or "loading" in image_src.lower():
                time.sleep(0.3)
            else:
                return captcha_image
        except Exception as e:
            print(f"Error: Captcha processing failed - {e}")
            return None

    print("Error: Failed to retrieve captcha after max attempts.")
    return None


def get_slider(driver):
    """
    Get slider element from webdriver
    """
    try:
        return WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "passMod_slide-btn"))
        )
    except Exception:
        print("Error: Slider not found.")
        return None


def calculate_drag_distance(pred_angle, total_moving_distance=238):
    """
    Calculate dragging distance
    """
    return int(((360 - pred_angle) / 360) * total_moving_distance)


def human_drag_curve(total_distance, steps=25):
    """
    Generate a human-like drag trajectory (easing function + random perturbation)
    """
    trajectory = []
    for i in range(steps):
        t = i / (steps - 1)
        ease_factor = t * (2 - t)

        stage_distance = total_distance * ease_factor - sum(trajectory)
        randomness = random.uniform(-0.2, 0.2) * stage_distance
        stage_distance += randomness

        trajectory.append(int(round(stage_distance)))
    return trajectory


def perform_drag(driver, slider, distance):
    """
    Perform drag on slider
    """
    action = ActionChains(driver, duration=50)
    action.click_and_hold(slider).pause(random.uniform(0.1, 0.3))

    trajectory = human_drag_curve(distance)
    y_offsets = [random.randint(-2, 2) for _ in trajectory]

    for dx, dy in zip(trajectory, y_offsets):
        action.pause(random.uniform(0.05, 0.15))
        action.move_by_offset(dx, dy)

    action.release()
    action.pause(random.uniform(0.3, 0.7))
    action.perform()


def solve_captcha(driver, model, transform, save_path=None):
    """
    Predict captcha angle and slide to verify
    """
    cap = find_cap(driver)
    if not cap:
        return None

    image_bytes = cap.screenshot_as_png

    if save_path:
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        timestamp = int(time.time() * 1000)
        filename = os.path.join(save_path, f"captcha_{timestamp}.png")
        with open(filename, "wb") as f:
            f.write(image_bytes)

    pred_angle = predict_angle(image_bytes, model, transform, input_type="bytes")
    if pred_angle is None:
        return None

    slider = get_slider(driver)
    if not slider:
        return None

    distance = calculate_drag_distance(pred_angle)
    perform_drag(driver, slider, distance)

    if "https://wappass.baidu.com/static/captcha/tuxing.html" not in driver.current_url:
        driver.get(TEST_PAGE_LINK)
        return True
    result_bar = driver.find_element(By.CLASS_NAME, "passMod_slide-grand")
    result_bar_class = result_bar.get_attribute("class")
    return "passMod_slide-grand-success" in result_bar_class


def batch_solve_captcha(driver, n, model, transform, save_path=None):
    """
    Batch solve captcha
    """
    success_count = 0
    with tqdm(total=n, desc="Solving captcha") as pbar:
        current_count = 0
        while current_count < n:
            result = solve_captcha(driver, model, transform, save_path=save_path)
            if result is True:
                success_count += 1
                current_count += 1
                pbar.update(1)
            elif result is False:
                current_count += 1
                pbar.update(1)
            pbar.set_postfix(pass_rate=f"{success_count / current_count * 100:.2f}%")
            driver.refresh()
    return success_count, n


def download_caps(driver, total_caps, output_dir, start_index=1):
    """
    Download multiple captcha images
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(start_index, total_caps + start_index), desc="Downloading captcha images"):
        cap = find_cap(driver)
        if cap:
            try:
                cap.screenshot(os.path.join(output_dir, f"{i}.png"))
            except Exception as e:
                print(f"Error: Failed to save image {i}.png - {e}")

    driver.quit()
