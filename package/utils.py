import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import torch


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_curve(x, y, curve_type):
    """
    Plots ROC or PRC
    :param x: The x co-ordinates
    :param y: The y co-ordinates
    :param curve_type: one of 'roc' or 'prc'
    :return: Plotly figure
    """
    auc = torch.sum(y * x) / torch.sum(x)  # 简单计算AUC
    x, y = x.numpy(), y.numpy()
    if curve_type == 'roc':
        title = f"ROC, AUC = {auc:.4f}"
        labels = dict(x='FPR', y='TPR')
    elif curve_type == 'prc':
        title = f"PRC, AUC = {auc:.4f}"
        labels = dict(x='Recall', y='Precision')
    else:
        raise ValueError(f"Invalid curve type - {curve_type}. Must be one of 'roc' or 'prc'.")

    # 创建图表
    fig = px.area(x=x, y=y, labels=labels, title=title)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    # 输出日志
    logger.info(f"{curve_type.upper()} curve generated with AUC = {auc:.4f}")

    return fig


def plot_confusion_matrix(
        cf,
        group_names=None,
        categories='auto',
        count=True,
        percent=True,
        cbar=True,
        xyticks=True,
        xyplotlabels=True,
        sum_stats=True,
        fig_size=None,
        cmap='Blues',
        title=None
):
    """
    Plot a confusion matrix with additional summary statistics.
    """
    plt.clf()

    blanks = ['' for _ in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = [f"{value}\n" for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = [f"{value:0.0f}\n" for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [f"{value:0.2%}\n" for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # 计算并显示额外统计
    if sum_stats:
        accuracy = np.trace(cf) / float(np.sum(cf))
        precision = cf[1, 1] / sum(cf[:, 1]) if len(cf) == 2 else None
        recall = cf[1, 1] / sum(cf[1, :]) if len(cf) == 2 else None
        f1_score = 2 * precision * recall / (precision + recall) if precision and recall else None
        stats_text = (
            f"\n\n"
            f"Accuracy={accuracy:.4f}\n"
            f"Precision={precision:.4f}\n"
            f"Recall={recall:.4f}\n"
            f"F1 Score={f1_score:.4f}"
        ) if f1_score else f"\n\nAccuracy={accuracy:.4f}"
    else:
        stats_text = ""

    # 图像参数
    if fig_size is None:
        fig_size = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False

    # 绘制热图
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
        xticklabels=categories, yticklabels=categories
    )

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig("CM.png")

    # 输出日志
    logger.info(f"Confusion matrix saved to CM.png.")
