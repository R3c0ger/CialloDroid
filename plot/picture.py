import matplotlib.pyplot as plt
import re


# 解析日志文件，提取epoch和f1 score
def parse_log(file_path):
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


# 日志文件路径
epochs_sage, f1_sage = parse_log('GraphConv_2layers_dropout.log')  
epochs_graph, f1_graph = parse_log('GraphConv_2layers_without_dropout.log')  


# 绘制折线图
plt.figure(figsize=(10, 6))

# 检查数据是否存在再绘图
if epochs_sage and f1_sage:
    plt.plot(epochs_sage, f1_sage, label="GraphConv_2layers with dropout", marker='o')
if epochs_graph and f1_graph:
    plt.plot(epochs_graph, f1_graph, label="GraphConv_2layers without dropout", marker='o')

# 设置图表标题和标签
plt.title('F1 Score Comparison: SAGEConv vs GraphConv')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# 显示图表
plt.grid(True)
plt.show()

