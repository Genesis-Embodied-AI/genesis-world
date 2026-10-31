#!/bin/bash

set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

uv venv --python '3.12' --clear /venv
source /venv/bin/activate
# Note: the version of cuda must tightly align with what is being installed
# in the Slurm container image, otherwise poorly packaged libraries, such as
# libuipc, may fail to import.
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install --upgrade pip setuptools wheel
uv pip install omniverse-kit --index-url https://pypi.nvidia.com/
uv pip install ".[dev,render,usd]" "pyuipc==0.0.7"
# FIXME: matplotlib 3.11.0 changed text rasterization, breaking the MPLPlotter PNG snapshot
# (test_recorders.py::test_plotter). Pin below 3.11 until the snapshot is regenerated.
uv pip install "matplotlib<3.11"
