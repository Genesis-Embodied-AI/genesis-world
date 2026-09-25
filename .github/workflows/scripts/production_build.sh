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
# imgui-bundle has no pre-built wheel for Python 3.10 (which this runner uses) so it is excluded from the
# ``[render]`` extras marker. Install it manually. Disable MicroTeX since we don't use it and its submodule
# is missing from the 1.92.x source distribution.
SKBUILD_CMAKE_DEFINE="IMGUI_BUNDLE_WITH_MICROTEX=OFF" uv pip install imgui-bundle
# FIXME: matplotlib 3.11.0 changed text rasterization, breaking the MPLPlotter PNG snapshot
# (test_recorders.py::test_plotter). Pin below 3.11 until the snapshot is regenerated.
uv pip install "matplotlib<3.11"
