from utils.crawler_utils import start_driver, download_caps


driver = start_driver(headless=True)
download_caps(driver, total_caps=10, output_dir="tmp")
