import matplotlib.pyplot as plt
import re


def parse_log(file_path):
    """解析日志文件，提取 epoch 和 f1 score"""
    epochs = []
    f1_scores = []

    with open(file_path, 'r') as f:
        for line in f:
            print("Processing line:", line.strip())  # 打印当前行
            # 查找包含 Epoch X Results 的行
            if 'Epoch' in line and 'Results' in line:
                print("Matched line:", line.strip())  # 打印匹配的行
                # 提取 F1 Score
                f1_match = re.search(r'F1 = ([\d.]+)', line)

                # 提取epoch
                epoch_match = re.search(r'Epoch (\d+) Results', line)
                epoch = int(epoch_match.group(1))  # 提取 epoch

                if f1_match and epoch_match:
                    epoch = int(epoch_match.group(1))  # 提取 epoch
                    f1_score = float(f1_match.group(1))  # 提取 F1 score
                    epochs.append(epoch)
                    f1_scores.append(f1_score)

    return epochs, f1_scores


def plot_f1_score(
        a_log_path: str,
        b_log_path: str,
        a_label: str,
        b_label: str,
        title: str = 'F1 Score Comparison',
        save_title: str = None
):
    """绘制 F1 Score 折线图"""
    # 日志文件路径
    epochs_a, f1_a = parse_log(a_log_path)
    epochs_b, f1_b = parse_log(b_log_path)

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    # 检查数据是否存在再绘图
    if epochs_a and f1_a:
        plt.plot(epochs_a, f1_a, label=a_label, marker='o')
    if epochs_b and f1_b:
        plt.plot(epochs_b, f1_b, label=b_label, marker='o')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    if not save_title:
        save_title = title
    plt.savefig(f'img/{save_title}.png', dpi=300)
    plt.savefig(f'img/{save_title}.svg', dpi=300)
    plt.show()


# GraphConv with and without dropout
plot_f1_score(
    'log/GraphConv_2layers_dropout.log', 
    'log/GraphConv_2layers_without_dropout.log', 
    'GraphConv_2layers with dropout', 
    'GraphConv_2layers without dropout',
    'F1 Score Comparison: GraphConv with and without dropout',
    "GraphConv"
)
# SAGEConv with and without dropout
plot_f1_score(
    'log/SAGEConv_2layers_dropout.log', 
    'log/SAGEConv_2layers_without_dropout.log', 
    'SAGEConv_2layers with dropout', 
    'SAGEConv_2layers without dropout',
    'F1 Score Comparison: SAGEConv with and without dropout',
    "SAGEConv"
)
# GraphConv and SAGEConv (with dropout)
plot_f1_score(
    'log/GraphConv_2layers_dropout.log', 
    'log/SAGEConv_2layers_dropout.log', 
    'GraphConv_2layers with dropout', 
    'SAGEConv_2layers with dropout',
    'F1 Score Comparison: GraphConv and SAGEConv (with dropout)',
    "GraphConv_vs_SAGEConv"
)
