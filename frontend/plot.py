#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import networkx as nx
import streamlit as st
from matplotlib import pyplot as plt
from pyvis.network import Network
from streamlit.components.v1 import html


def plt_plot(nx_graph, filename):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(nx_graph)  # 使用 spring 布局算法
    nx.draw(
        nx_graph, pos, ax=ax,
        with_labels=True,
        node_size=500,
        node_color='skyblue',
        font_size=10,
        font_weight='bold',
        edge_color='gray'
    )
    ax.set_title(f"APK Call Graph of {filename}")
    plt.show()
    st.pyplot(fig)


def plot_dgl2pyvis(dgl_graph, filename, show_in_st=True):
    # 创建 PyVis 网络对象
    net = Network(notebook=True, height='750px', width='100%')

    # 从 DGL 图中获取节点和边的信息
    nodes = dgl_graph.nodes().tolist()
    edges = dgl_graph.edges()

    # 将 DGL 图中的节点和边添加到 PyVis 图中
    for node in nodes:
        net.add_node(node, label=str(node))
    for src, dst in zip(edges[0].tolist(), edges[1].tolist()):
        net.add_edge(src, dst)

    # 保存 PyVis 图为 HTML 文件
    graph_dir = Path("tmp/graph")
    graph_dir.mkdir(parents=True, exist_ok=True)
    net.show(str(graph_dir / f"{filename}.html"))

    # 在 streamlit 中显示 PyVis 图
    if show_in_st:
        html(net.html, height=750)
