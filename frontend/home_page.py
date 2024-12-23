#!usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from style import heading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from main import VERSION
from predict import mal_detect


# 初始化 session state，记录已经检测的文件数
if 'predicted_num' not in st.session_state:
    st.session_state.predicted_num = 0

# 初始化结果数据
columns = ["filename", "size_kb", "prob", "is_mal", "is_apk"]
result_data = pd.DataFrame(columns=columns)

# APK 保存路径
apk_save_path = Path("tmp/apk")
apk_save_path.mkdir(parents=True, exist_ok=True)


# 主页面
heading("CialloDroid：基于图神经网络的安卓恶意软件检测模型", level=1)
github_url = "https://github.com/R3c0ger/CialloDroid"
st.write(
    f"![Version](https://img.shields.io/badge/version-v{VERSION}-ff4b4b?"
    f"style=for-the-badge)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
    f"[![GitHub](https://img.shields.io/badge/github-repo-ff4b4b?"
    f"style=for-the-badge&logo=github&logoColor=white)]({github_url})"
)

heading("上传文件")
st.write("在下方打开 APK 文件，或拖动 APK 文件到下方，将会自动进行检测。")
uploaded_file_list = st.file_uploader(
    "仅允许上传 APK 文件，一次可上传多个，每个文件大小不超过 200 MB。",
    type="apk",
    accept_multiple_files=True,
)
do_remain_data = st.checkbox("保留已检测文件的数据（本次上传文件数据清空，从下次上传开始保留）", value=True)


# 若有上传文件，则进行检测
if uploaded_file_list:
    dgl_graph = None

    # 显示检测进度条
    heading("检测进度")
    progress_bar = st.progress(0)
    apk_count = len(uploaded_file_list) - st.session_state.predicted_num
    apk_processed = 0

    for i, file in enumerate(uploaded_file_list):
        # 如果已经检测过，则跳过
        if i < st.session_state.predicted_num:
            continue

        # 初始化一个 apk 文件的检测结果
        result_row = {
            "filename": file.name,
            "size_kb": file.size / 1024,
            "prob": -1,
            "is_mal": "非 APK 文件",
            "is_apk": False,
        }

        # 根据文件头判断是否为 apk 文件
        file_bytes = file.read()
        file.seek(0)
        file_header = file_bytes[:4].hex()

        # 若是，则进行检测
        if file_header == "504b0304":
            result_row["is_apk"] = True
            # 将 apk 保存到 tmp 文件夹下
            apk_path = apk_save_path / file.name
            apk_path.write_bytes(file_bytes)
            # 进行检测
            result_row["prob"] = mal_detect(str(apk_path))
            result_row["is_mal"] = "恶意软件" if result_row["prob"] > 0.5 else "正常软件"

        # 更新结果数据
        result_data = pd.concat(
            [result_data, pd.DataFrame([result_row])],
            ignore_index=True
        )

        # 更新进度条
        apk_processed += 1
        progress_bar.progress(apk_processed / apk_count)

    # 输出检测结果
    heading("检测结果")
    st.dataframe(
        result_data,
        use_container_width=True,
        column_config={
            "filename": "文件名称",
            "size_kb": st.column_config.NumberColumn(
                "文件大小",
                format="%.2f KB",
            ),
            "is_mal": "检测结果",
            "prob": st.column_config.NumberColumn(
                "恶意软件概率",
                format="%.4f",
            ),
            "is_apk": "是否为 APK 文件",
        }
    )

    # 每次上传并检测完成后，重置进度条数据
    progress_bar = st.progress(0)
    apk_processed = 0

    # 更新 predicted_num
    if not do_remain_data:
        st.session_state.predicted_num = len(uploaded_file_list)
