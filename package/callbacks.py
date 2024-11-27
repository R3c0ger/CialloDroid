import logging
from typing import Tuple, List, Union

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch

from package.utils import plot_confusion_matrix, plot_curve


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")]
)
logger = logging.getLogger(__name__)


class InputMonitor:
    """
    Monitors and logs the histogram of input labels during training.
    """

    def __init__(self):
        pass

    @staticmethod
    def log_train_batch_start(batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor], batch_idx: int):
        """
        Logs histogram of labels for a training batch.
        :param batch: Tuple of graph data and labels.
        :param batch_idx: Index of the current batch.
        """
        _, labels = batch
        unique, counts = labels.unique(return_counts=True)
        label_distribution = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"Batch {batch_idx}: Label distribution: {label_distribution}")


class BestModelTagger:
    """
    Logs and tracks the best model based on a monitored metric (e.g., validation loss).
    """

    def __init__(self, monitor: str = 'val_loss', mode: str = 'min'):
        """
        :param monitor: Metric to monitor (e.g., 'val_loss').
        :param mode: One of 'min' or 'max'.
        """
        self.monitor = monitor
        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode {mode}. Must be one of 'min' or 'max'.")
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else -np.inf

    def update(self, current_score: float, epoch: int):
        """
        Updates the best score if the current score is better.
        :param current_score: Current value of the monitored metric.
        :param epoch: Current epoch number.
        """
        is_better = (current_score < self.best_score) if self.mode == 'min' else (current_score > self.best_score)
        if is_better:
            self.best_score = current_score
            logger.info(f"New best {self.monitor}: {self.best_score:.4f} at epoch {epoch}")


class MetricsLogger:
    """
    Logs and visualizes metrics for different stages (train, val, test).
    """

    def __init__(self, stages: Union[List[str], str]):
        """
        :param stages: List of stages to monitor (e.g., ['train', 'val', 'test']).
        """
        valid_stages = {'train', 'val', 'test'}
        if stages == 'all':
            self.stages = valid_stages
        else:
            for stage in stages:
                if stage not in valid_stages:
                    raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}.")
            self.stages = set(stages)

    @staticmethod
    def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Computes basic metrics like accuracy and F1 score.
        :param predictions: Model predictions.
        :param labels: Ground-truth labels.
        :return: Dictionary of metrics.
        """
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return {'accuracy': accuracy}

    @staticmethod
    def log_metrics(metrics: dict, stage: str):
        """
        Logs computed metrics for a specific stage.
        :param metrics: Dictionary of metrics.
        :param stage: Stage name (e.g., 'train', 'val', 'test').
        """
        logger.info(f"{stage.capitalize()} Metrics: {metrics}")

    @staticmethod
    def log_confusion_matrix(confusion_matrix: np.ndarray, stage: str):
        """
        Logs and plots the confusion matrix.
        :param confusion_matrix: Confusion matrix as a NumPy array.
        :param stage: Stage name (e.g., 'test').
        """
        plot_confusion_matrix(
            confusion_matrix,
            group_names=['TN', 'FP', 'FN', 'TP'],
            categories=['Benign', 'Malware'],
            cmap='binary'
        )
        plt.savefig(f"{stage}_confusion_matrix.png")
        logger.info(f"Saved {stage} confusion matrix as {stage}_confusion_matrix.png")

    @staticmethod
    def log_roc_curve(roc_data: Tuple[np.ndarray, np.ndarray], stage: str):
        """
        Logs and plots the ROC curve.
        :param roc_data: Tuple of (FPR, TPR).
        :param stage: Stage name (e.g., 'test').
        """
        figure = plot_curve(roc_data[0], roc_data[1], 'roc')
        plt.savefig(f"{stage}_roc_curve.png")
        logger.info(f"Saved {stage} ROC curve as {stage}_roc_curve.png")

    @staticmethod
    def log_prc_curve(prc_data: Tuple[np.ndarray, np.ndarray], stage: str):
        """
        Logs and plots the Precision-Recall curve.
        :param prc_data: Tuple of (Precision, Recall).
        :param stage: Stage name (e.g., 'test').
        """
        figure = plot_curve(prc_data[1], prc_data[0], 'prc')
        plt.savefig(f"{stage}_prc_curve.png")
        logger.info(f"Saved {stage} PRC curve as {stage}_prc_curve.png")
