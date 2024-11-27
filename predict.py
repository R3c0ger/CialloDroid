from pathlib import Path

import dgl
import torch

from package.model import MalwareDetector
from package.process_dataset import process_apk


# 配置参数和模型
CONFIG = {
    "input_dimension": 253,
    "convolution_algorithm": "SAGEConv",
    "convolution_count": 2,  # 修改为与训练时相同的卷积层数
    "model_path": "checkpoints/epoch_99_val_loss_5.4032.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def load_model(config):
    model = MalwareDetector(
        input_dimension=config["input_dimension"],
        convolution_algorithm=config["convolution_algorithm"],
        convolution_count=config["convolution_count"]
    )
    model.load_state_dict(torch.load(config["model_path"], map_location=config["device"]), strict=False)
    model.to(config["device"])
    model.eval()
    return model


def process_apk_file(apk_path, processed_dir):
    """处理 APK 文件，生成调用图"""
    process_apk(apk_path, processed_dir, label=-1)  # 使用 `label=-1`，仅用于预测，不需要具体标签
    graph_path = processed_dir / f"{apk_path.stem}.fcg"
    return graph_path


def predict(graph_path, model, device):
    """加载图并进行预测"""
    graphs, _ = dgl.data.utils.load_graphs(str(graph_path))
    graph = graphs[0]
    dgl_graph = graph

    # 合并多个特征为 'features'，确保特征与训练时一致
    feature_keys = ['api', 'user', 'external', 'entrypoint', 'native', 'public', 'static', 'codesize']
    features = []
    for key in feature_keys:
        if key in graph.ndata:
            data = graph.ndata[key]
            # 如果特征是一维张量，将其调整为二维张量
            if data.dim() == 1:
                data = data.unsqueeze(-1)
            features.append(data)
    if features:
        all_features = torch.cat(features, dim=1)
    else:
        raise ValueError("No valid features found for graph.")

    graph.ndata['features'] = all_features.float()  # 确保节点特征是浮点数
    graph = graph.to(device)

    with torch.no_grad():
        output = model(graph)
        prob = torch.sigmoid(output).item()

    return prob, dgl_graph


def mal_detect(apk_path_str):
    apk_path = Path(apk_path_str)
    processed_dir = Path("tmp/processed_fcg")
    processed_dir.mkdir(exist_ok=True)

    # 加载模型
    model = load_model(CONFIG)

    # 处理 APK 文件
    graph_path = process_apk_file(apk_path, processed_dir)

    # 进行预测
    prob, dgl_graph = predict(graph_path, model, CONFIG["device"])

    # 打印预测结果
    if prob > 0.5:
        print(f"The APK {apk_path} is predicted to be MALICIOUS with probability {prob:.4f}")
    else:
        print(f"The APK {apk_path} is predicted to be BENIGN with probability {prob:.4f}")

    return prob, dgl_graph


if __name__ == "__main__":
    """
    使用示例：
    python predict.py data/benign/Benigh3135.apk
    """
    import argparse

    parser = argparse.ArgumentParser(description="Predict whether an APK file is malicious.")
    parser.add_argument("apk_path", type=str, help="Path to the APK file to be predicted.")
    args = parser.parse_args()
    mal_detect(args.apk_path)
