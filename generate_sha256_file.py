import hashlib
import os


# 文件夹路径
data_dir = "./data"
benign_dir = os.path.join(data_dir, "benign")
malicious_dir = os.path.join(data_dir, "malicious")

# 输出文件
benign_sha256_file = "./data/benign_data.sha256"
malicious_sha256_file = "./data/malicious_data.sha256"


def compute_sha256(file_path):
    """计算文件的 SHA256 值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_directory(directory, output_file):
    """处理指定目录中的所有文件，生成 sha256 列表并保存"""
    with open(output_file, "w") as f:
        for file_name in sorted(os.listdir(directory)):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                sha256_hash = compute_sha256(file_path)
                f.write(f"{sha256_hash}  {file_name}\n")


# 生成 benign 和 malicious 数据的 SHA256 文件
process_directory(benign_dir, benign_sha256_file)
process_directory(malicious_dir, malicious_sha256_file)
