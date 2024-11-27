#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st


title_style = """
    font-size: 2.15rem;
    font-weight: 700;
    padding: 1.25rem 0px 1rem;
    margin: 0px;
    line-height: 1.2;
"""

subtitle_style = """
    font-size: 1.5rem;
    font-weight: 600;
    padding: 0px 0px 1rem;
    margin: 0px;
    line-height: 1.2;
"""


def heading(text: str, level: int = 2, style: str = ""):
    if not style:
        if level == 1:
            style = title_style
        elif level == 2:
            style = subtitle_style
    st.markdown(
        f"<h{level} style='{style}'>{text}</h{level}>",
        unsafe_allow_html=True,
    )
