"""Record the environment used for an external compiler benchmark."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    mode, image_version = sys.argv[1:]
    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower().replace("_", "-") not in ("genesis-world", "quadrants")
    )
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
    )
    environment = dict(
        image_version=image_version,
        python=platform.python_version(),
        hardware=sorted(set(hardware.splitlines())),
        packages=packages,
        build_script=hashlib.sha256(Path(__file__).with_name("production_build.sh").read_bytes()).hexdigest(),
    )
    Path(f"benchmark-environment-{mode}.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
