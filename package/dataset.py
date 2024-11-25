from pathlib import Path
from typing import List, Dict, Tuple, Union

import dgl
import torch
from torch.utils.data import Dataset

attributes = {'external', 'entrypoint', 'native', 'public', 'static', 'codesize'}


class MalwareDataset(Dataset):
    """
    数据集类，用于加载处理后的图数据和标签。
    """

    def __init__(self, graph_files: List[Path], consider_features: List[str]):
        """
        :param graph_files: 包含 `.fcg` 图数据文件的路径列表。
        :param consider_features: 需要从图节点属性中考虑的特征。
        """
        self.graph_files = graph_files
        self.consider_features = consider_features

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        返回单个图和其全局标签。
        """
        graph_path = self.graph_files[idx]
        graphs, _ = dgl.data.utils.load_graphs(str(graph_path))
        graph = graphs[0]

        # 动态生成 features
        feature_list = []
        for feature in self.consider_features:
            if feature in graph.ndata:
                feature_list.append(graph.ndata[feature].float().unsqueeze(-1))
        if feature_list:
            graph.ndata['features'] = torch.cat(feature_list, dim=1)
        else:
            raise ValueError(f"No valid features found for graph {graph_path.stem}")

        label = graph.ndata['label'][0]  # 假设所有节点的标签相同
        return graph, label

