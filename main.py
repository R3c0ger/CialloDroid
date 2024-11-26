#!usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys


VERSION = "0.1.0"


if __name__ == "__main__":
    subprocess.run(["streamlit", "run", r".\frontend\home_page.py"])
    sys.exit(0)
