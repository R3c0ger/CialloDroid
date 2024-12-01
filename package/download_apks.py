import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


# 切换到文件所在目录
current_dir = Path(__file__).resolve().parent
os.chdir(current_dir)


def read_api_key():
    """从 api.conf 中读取 API Key"""
    api_conf_path = current_dir / "../api.conf"
    with api_conf_path.open("r") as file:
        api_key = file.read().strip()
        print(f"API Key: {api_key}")
        return api_key


# 配置参数
API_KEY = read_api_key()  # AndroZoo API Key
DATA_FOLDER = current_dir / "../data"  # 数据存放目录
INPUT_FILE = DATA_FOLDER / "train_old.sha256"  # 包含 SHA256 和文件名的文件
BENIGN_DIR = DATA_FOLDER / "benign"  # 正常软件存放目录
MALICIOUS_DIR = DATA_FOLDER / "malicious"  # 恶意软件存放目录
BASE_URL = "https://androzoo.uni.lu/api/download"
MAX_WORKERS = 20  # 最大并发下载数

# 创建存储目录
BENIGN_DIR.mkdir(parents=True, exist_ok=True)
MALICIOUS_DIR.mkdir(parents=True, exist_ok=True)

# 读取 SHA256 列表
with INPUT_FILE.open("r") as file:
    lines = file.readlines()


# 下载函数
def download_apk(line):
    try:
        sha256, apk_name = line.strip().split(maxsplit=1)
        # 确定保存路径
        if "benigh" in apk_name.lower():
            output_path = BENIGN_DIR / apk_name
        else:
            output_path = MALICIOUS_DIR / apk_name

        # 检查是否已下载
        if output_path.exists():
            print(f"File already exists: {output_path}")
            return

        # 下载文件
        url = f"{BASE_URL}?apikey={API_KEY}&sha256={sha256}"
        print(f"Downloading {apk_name} from {url}")
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with output_path.open("wb") as apk_file:
                for chunk in response.iter_content(chunk_size=8192):
                    apk_file.write(chunk)
            print(f"Saved {apk_name} to {output_path}")
        elif response.status_code == 404:
            print(f"File not found for SHA256: {sha256}, skipping...")
        else:
            print(f"Failed to download {apk_name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error processing {line.strip()}: {e}")


# 并发下载
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_apk, lines)
