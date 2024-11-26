#!usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import streamlit as st
# import torch

from style import title_style, subtitle_style, heading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# from package.model import MalwareDetector
from main import VERSION


heading("CialloDroid：基于图神经网络的安卓恶意软件检测模型", level=1)
st.write(f"Version: {VERSION}")

heading("上传文件")
st.write("在下方打开 APK 文件，或拖动 APK 文件到下方，将会自动进行检测。")
uploaded_file_list = st.file_uploader(
    "仅允许上传 APK 文件，一次可上传多个，每个文件大小不超过 200 MB。",
    type="apk",
    accept_multiple_files=True,
)

columns = ["filename", "size_kb", "is_mal", "is_apk"]
result_data = pd.DataFrame(columns=columns)


# 若有上传文件，则进行检测
if uploaded_file_list:
    # 创建 tmp 文件夹用于存放上传的 apk 文件
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    # 显示检测进度条
    heading("检测进度")
    progress_bar = st.progress(0)
    apk_count = len(uploaded_file_list)
    apk_processed = 0

    # model = MalwareDetector(input_dimension=253, convolution_count=2, convolution_algorithm="GraphConv")
    # model.load_state_dict(torch.load('../checkpoints/best_model.pt'))
    # model.eval()  # 设置模型为评估模式

    for file in uploaded_file_list:
        # 初始化一个 apk 文件的检测结果
        result_row = {
            "filename": file.name,
            "size_kb": file.size / 1024,
            "is_mal": "非 APK 文件，无法检测",
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
            with open(f"tmp/{file.name}", "wb") as f:
                f.write(file_bytes)
            # TODO: 以下为检测部分
            pass
            # TODO: 以上为检测部分

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
            "is_apk": "是否为 APK 文件",
        }
    )

    # 每次上传并检测完成后，重置进度条数据
    progress_bar = st.progress(0)
    apk_count = len(uploaded_file_list)
    apk_processed = 0
