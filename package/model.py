from typing import Mapping, Tuple, Optional, Dict

import dgl
import dgl.nn.pytorch as graph_nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList


class MalwareDetector(nn.Module):
    def __init__(
            self,
            input_dimension: int,
            convolution_algorithm: str,
            convolution_count: int,
    ):
        super().__init__()
        supported_algorithms = ['GraphConv', 'SAGEConv', 'TAGConv', 'DotGatConv']
        if convolution_algorithm not in supported_algorithms:
            raise ValueError(
                f"{convolution_algorithm} is not supported. Supported algorithms are {supported_algorithms}")
        self.convolution_layers = ModuleList()  # 使用 ModuleList
        convolution_dimensions = [64, 32, 16]
        for dimension in convolution_dimensions[:convolution_count]:
            self.convolution_layers.append(self._get_convolution_layer(
                name=convolution_algorithm,
                input_dimension=input_dimension,
                output_dimension=dimension
            ))
            input_dimension = dimension
        self.classify = nn.Linear(input_dimension, 1)  # 最终分类层

    @staticmethod
    def _get_convolution_layer(
            name: str,
            input_dimension: int,
            output_dimension: int
    ) -> torch.nn.Module:
        return {
            "GraphConv": graph_nn.GraphConv(
                input_dimension,
                output_dimension,
                activation=F.relu,
                allow_zero_in_degree=True  # 允许 0 入度节点
            ),
            "SAGEConv": graph_nn.SAGEConv(
                input_dimension,
                output_dimension,
                activation=F.relu,
                aggregator_type='mean',
                norm=F.normalize
            ),
            "DotGatConv": graph_nn.DotGatConv(
                input_dimension,
                output_dimension,
                num_heads=1
            ),
            "TAGConv": graph_nn.TAGConv(
                input_dimension,
                output_dimension,
                k=4
            )
        }.get(name, None)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        h = g.ndata['features']  # 获取节点特征
        for layer in self.convolution_layers:
            h = layer(g, h)  # 逐层调用卷积层
        g.ndata['h'] = h  # 保存节点嵌入
        hg = dgl.mean_nodes(g, 'h', ntype=None)  # 使用平均池化获取每个图的特征
        return self.classify(hg).squeeze(-1)  # 确保输出为 [batch_size]

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes binary cross-entropy loss.
        """
        return self.loss_func(logits, labels)

    def evaluate_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate all metrics based on predictions and ground-truth labels.
        """
        predictions = torch.sigmoid(logits).round()
        results = {name: metric(predictions, labels) for name, metric in self.metrics.items()}
        return results

    @staticmethod
    def _accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        return (predictions == labels).float().mean().item()

    @staticmethod
    def _precision(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        return tp / (tp + fp + 1e-10)

    @staticmethod
    def _recall(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        return tp / (tp + fn + 1e-10)

    @staticmethod
    def _f1_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        precision = MalwareDetector._precision(predictions, labels)
        recall = MalwareDetector._recall(predictions, labels)
        return 2 * (precision * recall) / (precision + recall + 1e-10)
