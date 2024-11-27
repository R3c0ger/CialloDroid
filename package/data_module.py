from pathlib import Path
from typing import List, Tuple, Union

import dgl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader


def stratified_split_dataset(
    samples: List[Path], labels: List[int], train_ratio: float = 0.8
) -> Tuple[List[Path], List[Path]]:
    """
    分层划分数据集为训练集和验证集。
    默认按 8:2 比例划分。
    """
    val_ratio = 1 - train_ratio  # 根据 train_ratio 自动计算 val_ratio
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(samples, labels))
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


class MalwareDataset(Dataset):
    """数据集类，用于加载处理后的图数据和标签。"""

    def __init__(self, graph_files: List[Path], consider_features: List[str]):
        """
        :param graph_files: 包含 `.fcg` 图数据文件的路径列表。
        :param consider_features: 要考虑的节点特征列表。
        """
        self.graph_files = graph_files
        self.consider_features = consider_features

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """返回单个图和其全局标签。"""
        graph_path = self.graph_files[idx]
        graphs, _ = dgl.data.utils.load_graphs(str(graph_path))
        graph = graphs[0]

        # 筛选并构建节点特征
        features = []
        for feature in self.consider_features:
            if feature in graph.ndata:
                data = graph.ndata[feature]
                # 如果特征是一维张量，将其调整为二维张量
                if data.dim() == 1:
                    data = data.unsqueeze(-1)
                features.append(data)
            else:
                raise KeyError(f"Feature '{feature}' not found in graph node data.")

        if features:
            graph.ndata['features'] = torch.cat(features, dim=1)
        else:
            raise ValueError("No valid features found for graph.")

        # 假设所有节点的标签相同
        label = graph.ndata['label'][0]
        return graph, label


class DataModule:
    def __init__(
            self,
            train_dir: Union[str, Path],
            test_dir: Union[str, Path],
            batch_size: int,
            split_ratios: Tuple[float, float],
            num_workers: int,
            pin_memory: bool,
            consider_features: List[str],
            split_train_val: bool = True,
    ):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_train_val = split_train_val
        self.consider_features = consider_features

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None

    def setup(self):
        train_files, train_labels = self._get_samples(self.train_dir)
        test_files, test_labels = self._get_samples(self.test_dir)

        if self.split_train_val:
            train_files, val_files = stratified_split_dataset(train_files, train_labels, self.split_ratios[0])
            self.val_dataset = MalwareDataset(val_files, self.consider_features)
        else:
            self.val_dataset = MalwareDataset(test_files, self.consider_features)

        self.train_dataset = MalwareDataset(train_files, self.consider_features)
        self.test_dataset = MalwareDataset(test_files, self.consider_features)

    @staticmethod
    def _get_samples(data_dir: Path) -> Tuple[List[Path], List[int]]:
        """从指定目录中加载样本文件路径和标签"""
        graph_files = sorted(data_dir.glob("*.fcg"))
        if not graph_files:
            raise FileNotFoundError(f"No .fcg files found in directory: {data_dir}")
        labels = []
        for graph_file in graph_files:
            graphs, _ = dgl.data.utils.load_graphs(str(graph_file))
            labels.append(graphs[0].ndata["label"][0].item())  # 假设所有节点的标签相同
        return graph_files, labels

    def get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        graphs, labels = zip(*samples)
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels).squeeze(-1)  # 将 labels 变为一维
        return batched_graph, labels

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_dataset, shuffle=False)
