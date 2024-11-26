import os
import traceback
from pathlib import Path
import torch
import dgl
import networkx as nx
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import MethodAnalysis, Analysis
from pygtrie import StringTrie
from collections import defaultdict
from typing import Dict, List, Union, Optional
# from androguard.util import set_log
# set_log("ERROR")

# import inspect
# print(inspect.getfile(Analysis))
# print(inspect.getsource(Analysis))

# 参数配置
DATA_DIR = "../data"  # 存放所有 APK 文件（恶意和正常）的目录
PROCESSED_DIR = "../processed_data"  # 存放处理后数据的目录
NUM_JOBS = 4  # 并行处理的线程数
OVERRIDE_EXISTING = True  # 是否覆盖已处理文件

# 节点属性
ATTRIBUTES = ['external', 'entrypoint', 'native', 'public', 'static', 'codesize', 'api', 'user']

stats: Dict[str, int] = defaultdict(int)


def memoize(function):
    """
    Alternative to @lru_cache which could not be pickled in ray
    :param function: Function to be cached
    :return: Wrapped function
    """
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


class FeatureExtractors:
    NUM_PERMISSION_GROUPS = 20
    NUM_API_PACKAGES = 226
    NUM_OPCODE_MAPPINGS = 21

    @staticmethod
    def _get_opcode_mapping() -> Dict[str, int]:
        """
        Group opcodes and assign them an ID
        :return: Mapping from opcode group name to their ID
        """
        mapping = {x: i for i, x in enumerate(['nop', 'mov', 'return',
                                               'const', 'monitor', 'check-cast', 'instanceof', 'new',
                                               'fill', 'throw', 'goto/switch', 'cmp', 'if', 'unused',
                                               'arrayop', 'instanceop', 'staticop', 'invoke',
                                               'unaryop', 'binop', 'inline'])}
        mapping['invalid'] = -1
        return mapping

    @staticmethod
    @memoize
    def _get_instruction_type(op_value: int) -> str:
        """
        Get instruction group name from instruction
        :param op_value: Opcode value
        :return: String containing ID of :instr:
        """
        if 0x00 == op_value:
            return 'nop'
        elif 0x01 <= op_value <= 0x0D:
            return 'mov'
        elif 0x0E <= op_value <= 0x11:
            return 'return'
        elif 0x12 <= op_value <= 0x1C:
            return 'const'
        elif 0x1D <= op_value <= 0x1E:
            return 'monitor'
        elif 0x1F == op_value:
            return 'check-cast'
        elif 0x20 == op_value:
            return 'instanceof'
        elif 0x22 <= op_value <= 0x23:
            return 'new'
        elif 0x24 <= op_value <= 0x26:
            return 'fill'
        elif 0x27 == op_value:
            return 'throw'
        elif 0x28 <= op_value <= 0x2C:
            return 'goto/switch'
        elif 0x2D <= op_value <= 0x31:
            return 'cmp'
        elif 0x32 <= op_value <= 0x3D:
            return 'if'
        elif (0x3E <= op_value <= 0x43) or (op_value == 0x73) or (0x79 <= op_value <= 0x7A) or (
                0xE3 <= op_value <= 0xED):
            return 'unused'
        elif (0x44 <= op_value <= 0x51) or (op_value == 0x21):
            return 'arrayop'
        elif (0x52 <= op_value <= 0x5F) or (0xF2 <= op_value <= 0xF7):
            return 'instanceop'
        elif 0x60 <= op_value <= 0x6D:
            return 'staticop'
        elif (0x6E <= op_value <= 0x72) or (0x74 <= op_value <= 0x78) or (0xF0 == op_value) or (
                0xF8 <= op_value <= 0xFB):
            return 'invoke'
        elif 0x7B <= op_value <= 0x8F:
            return 'unaryop'
        elif 0x90 <= op_value <= 0xE2:
            return 'binop'
        elif 0xEE == op_value:
            return 'inline'
        else:
            return 'invalid'

    @staticmethod
    def _mapping_to_bitstring(mapping: List[int], max_len) -> torch.Tensor:
        """
        Convert opcode mappings to bitstring
        :param max_len: Length of the bitstring
        :param mapping: List of IDs of opcode groups (present in a method)
        :return: Binary tensor of length `len(opcode_mapping)` with value 1 at positions specified by :param mapping:
        """
        size = torch.Size([1, max_len])
        if len(mapping) > 0:
            indices = torch.LongTensor([[0, x] for x in mapping]).t()
            values = torch.LongTensor([1] * len(mapping))
            tensor = torch.sparse_coo_tensor(indices, values, size, dtype=torch.long)  # Updated to sparse_coo_tensor
        else:
            tensor = torch.sparse_coo_tensor(size, dtype=torch.long)  # Updated to sparse_coo_tensor
        return tensor.to_dense().squeeze()

    @staticmethod
    def _get_api_trie() -> StringTrie:
        apis = open('../metadata/api.list').readlines()
        api_list = {x.strip(): i for i, x in enumerate(apis)}
        api_trie = StringTrie(separator='.')
        for k, v in api_list.items():
            api_trie[k] = v
        return api_trie

    @staticmethod
    @memoize
    def get_api_features(api: MethodAnalysis) -> Optional[torch.Tensor]:
        if not api.is_external():
            return None
        api_trie = FeatureExtractors._get_api_trie()
        name = str(api.class_name)[1:-1].replace('/', '.')
        _, index = api_trie.longest_prefix(name)
        if index is None:
            indices = []
        else:
            indices = [index]
        feature_vector = FeatureExtractors._mapping_to_bitstring(indices, FeatureExtractors.NUM_API_PACKAGES)
        return feature_vector

    @staticmethod
    @memoize
    def get_user_features(user: MethodAnalysis) -> Optional[torch.Tensor]:
        if user.is_external():
            return None
        opcode_mapping = FeatureExtractors._get_opcode_mapping()
        opcode_groups = set()
        for instr in user.get_method().get_instructions():
            instruction_type = FeatureExtractors._get_instruction_type(instr.get_op_value())
            instruction_id = opcode_mapping[instruction_type]
            if instruction_id >= 0:
                opcode_groups.add(instruction_id)
        # 1 subtraction for 'invalid' opcode group
        feature_vector = FeatureExtractors._mapping_to_bitstring(list(opcode_groups), len(opcode_mapping) - 1)
        return torch.LongTensor(feature_vector)


def process_apk(apk_path: Path, dest_dir: Path, label: int):
    """
    处理单个 APK 文件，提取调用图并保存为 DGL 图
    :param apk_path: Path to the APK file
    :param dest_dir: Directory to save the processed graph
    :param label: 0 for benign, 1 for malicious
    """
    try:
        print(f"Processing {apk_path}")
        _, _, dx = AnalyzeAPK(apk_path)

        # 初始化 NetworkX 图
        cg = dx.get_call_graph()

        # 准备节点特征映射
        mappings = {}
        for node in cg.nodes():
            features = {
                "api": torch.zeros(FeatureExtractors.NUM_API_PACKAGES),
                "user": torch.zeros(FeatureExtractors.NUM_OPCODE_MAPPINGS)
            }
            if node.is_external():
                features["api"] = FeatureExtractors.get_api_features(node)
            else:
                features["user"] = FeatureExtractors.get_user_features(node)
            mappings[node] = features

        # 设置节点属性
        nx.set_node_attributes(cg, mappings)
        cg = nx.convert_node_labels_to_integers(cg)

        # 转换为 DGL 图
        dg = dgl.from_networkx(cg, node_attrs=ATTRIBUTES)

        # 为每个节点设置标签
        num_nodes = dg.num_nodes()
        dg.ndata['label'] = torch.full((num_nodes,), label, dtype=torch.long)

        # 保存处理结果
        dest_path = dest_dir / f"{apk_path.stem}.fcg"
        dgl.data.utils.save_graphs(str(dest_path), [dg])
        print(f"Saved processed graph to {dest_path}")
    except Exception as e:
        print(f"Error processing {apk_path}: {e}")
        traceback.print_exc()



def main():
    # 创建目标目录
    dest_dir = Path(PROCESSED_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 处理恶意软件
    malicious_dir = Path(DATA_DIR) / "malicious"
    malicious_apks = list(malicious_dir.glob("*.apk"))
    print(f"Found {len(malicious_apks)} malicious APKs")
    for apk in malicious_apks:
        process_apk(apk, dest_dir, label=1)  # 标签为 1 表示恶意

    # 处理正常软件
    benign_dir = Path(DATA_DIR) / "benign"
    benign_apks = list(benign_dir.glob("*.apk"))
    print(f"Found {len(benign_apks)} benign APKs")
    for apk in benign_apks:
        process_apk(apk, dest_dir, label=0)  # 标签为 0 表示正常

    print("All APKs processed.")


if __name__ == "__main__":
    main()
