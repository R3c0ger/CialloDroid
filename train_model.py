import logging
from pathlib import Path

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from package.data_module import DataModule
from package.model import MalwareDetector


# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 参数配置
CONFIG = {
    "data": {
        "train_dir": "./processed_data",
        "test_dir": "./processed_data",
        "batch_size": 32,
        "split_ratios": (0.8, 0.2),
        "consider_features": ['api', 'user', 'external', 'entrypoint', 'native', 'public', 'static', 'codesize'],
        "num_workers": 4,
        "pin_memory": True,
        "split_train_val": True,
    },
    "model": {
        "input_dimension": 253,  # 对应 consider_features 的长度
        "convolution_algorithm": "GraphConv",
        "convolution_count": 2,
    },
    "train": {
        "epochs": 20,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "./checkpoints",
    }
}


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        graphs, labels = batch
        graphs, labels = graphs.to(device), labels.to(device).float()  # 转换 labels 为 float

        optimizer.zero_grad()
        outputs = model(graphs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()  # 转换为 long 类型以比较
        total += labels.size(0)

    accuracy = correct / total
    logger.info(f"Train Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return epoch_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            graphs, labels = batch
            graphs, labels = graphs.to(device), labels.to(device).float()  # 转换 labels 为 float

            outputs = model(graphs)
            loss = criterion(outputs, labels)

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()  # 转换为 long 类型以比较
            total += labels.size(0)

    accuracy = correct / total
    logger.info(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return epoch_loss, accuracy


def main():
    # 初始化数据模块
    logger.info("Initializing data module...")
    data_module = DataModule(**CONFIG["data"])
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 初始化模型
    logger.info("Initializing model...")
    model = MalwareDetector(**CONFIG["model"])
    device = CONFIG["train"]["device"]
    model.to(device)

    # 优化器和损失函数
    optimizer = Adam(model.parameters(), lr=CONFIG["train"]["learning_rate"])
    criterion = BCEWithLogitsLoss()

    # 输出目录
    output_dir = Path(CONFIG["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练
    logger.info("Starting training...")
    best_val_loss = float("inf")
    best_checkpoint_path = None
    for epoch in range(1, CONFIG["train"]["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{CONFIG['train']['epochs']}")

        # 训练和验证
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 保存模型
        checkpoint_path = output_dir / f"epoch_{epoch:02d}_val_loss_{val_loss:.4f}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_checkpoint_path)

    logger.info(f"Training complete. Best model saved at: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
