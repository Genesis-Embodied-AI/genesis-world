import os
import re
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from functools import cache
from pathlib import Path

from httpcore import TimeoutException as HTTPTimeoutException
from httpx import HTTPError as HTTPXError
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError

from genesis.constants import GLTF_FORMATS, MESH_FORMATS, MJCF_FORMAT, URDF_FORMAT, USD_FORMATS
from genesis.recorders.trajectory import TRAJECTORY_FORMAT

REPOSITY_URL = "Genesis-Embodied-AI/Genesis"
DEFAULT_BRANCH_NAME = "main"

HUGGINGFACE_ASSETS_REVISION = "990a727788f11e34ad006c69bf769303b20cb11c"
HUGGINGFACE_SNAPSHOT_REVISION = "70384ecdce390b4088770e29666a454e795f5663"

MESH_EXTENSIONS = (".mtl", *MESH_FORMATS, *GLTF_FORMATS, *USD_FORMATS)
IMAGE_EXTENSIONS = (".png", ".jpg")

# Get repository "root" path (actually test dir is good enough)
TEST_DIR = os.path.dirname(os.path.dirname(__file__))


@cache
def get_git_commit_timestamp(ref="HEAD"):
    try:
        contrib_date = subprocess.check_output(
            ["git", "show", "-s", "--quiet", "--format=%ci", ref], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        # Commit not found, either because it does not exist or becaused fo shallow git clone
        return float("nan")

    try:
        date = datetime.fromisoformat(contrib_date)
    except ValueError:
        date = datetime.strptime(contrib_date, "%Y-%m-%d %H:%M:%S %z")
    timestamp = date.timestamp()

    return timestamp


@cache
def get_git_commit_info(ref="HEAD"):
    # Fetch current commit revision
    try:
        revision = subprocess.check_output(["git", "rev-parse", ref], cwd=TEST_DIR, encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        revision = f"{uuid.uuid4().hex}@UNKNOWN"
        timestamp = float("nan")
        return revision, timestamp

    # Fetch all remote branches containing the current commit
    try:
        branches = subprocess.check_output(
            ["git", "branch", "--remote", "--contains", ref], cwd=TEST_DIR, encoding="utf-8"
        ).splitlines()
    except subprocess.CalledProcessError:
        # Raise error if not found neither locally nor remotely
        branches = ()

    # Check if the current commit is contained by main branch
    remote_handle = "UNKNOWN"
    for branch in branches:
        try:
            remote_name, branch_name = branch.strip().split("/", 1)
        except ValueError:
            continue
        if branch_name != DEFAULT_BRANCH_NAME:
            continue
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", remote_name], cwd=TEST_DIR, encoding="utf-8"
        ).strip()
        try:
            remote_handle = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote_url).group(1)
        except AttributeError:
            pass
        if remote_handle == REPOSITY_URL:
            is_commit_on_default_branch = True
            break
    else:
        is_commit_on_default_branch = False
    revision = f"{revision}@{remote_handle}"

    # Return the contribution date as timestamp if and only if the HEAD commit is on main branch
    if is_commit_on_default_branch:
        timestamp = get_git_commit_timestamp(ref)
    else:
        timestamp = float("nan")

    return revision, timestamp


def get_hf_dataset(
    pattern,
    repo_name: str = "assets",
    local_dir: str | None = None,
    num_retry: int = 4,
    retry_delay: float = 30.0,
):
    assert num_retry >= 1

    if repo_name == "assets":
        revision = HUGGINGFACE_ASSETS_REVISION
    elif repo_name == "snapshots":
        revision = HUGGINGFACE_SNAPSHOT_REVISION
    else:
        raise ValueError(f"Unsupported repository '{repo_name}'")

    for i in range(num_retry):
        try:
            # Try downloading the assets
            asset_path = snapshot_download(
                repo_type="dataset",
                repo_id=f"Genesis-Intelligence/{repo_name}",
                revision=revision,
                allow_patterns=pattern,
                max_workers=1,
                local_dir=local_dir,
            )

            # Make sure that download was successful
            has_files = False
            for path in Path(asset_path).glob(pattern):
                if not path.is_file():
                    continue

                ext = path.suffix.lower()
                if ext not in (URDF_FORMAT, MJCF_FORMAT, TRAJECTORY_FORMAT, *IMAGE_EXTENSIONS, *MESH_EXTENSIONS):
                    continue

                has_files = True

                if path.stat().st_size == 0:
                    raise HTTPError(f"File '{path}' is empty.")

                if path.suffix.lower() in (URDF_FORMAT, MJCF_FORMAT):
                    try:
                        ET.parse(path)
                    except ET.ParseError as e:
                        raise HTTPError("Impossible to parse XML file.") from e
                elif path.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        Image.open(path)
                    except UnidentifiedImageError as e:
                        raise HTTPError("Impossible to parse Image file.") from e
                elif path.suffix.lower() in MESH_EXTENSIONS:
                    # TODO: Validating mesh files is more tricky. Ignoring them for now.
                    pass

            if not has_files:
                raise HTTPError("No file downloaded.")
        except (HTTPTimeoutException, HTTPXError, HTTPError, LocalEntryNotFoundError, FileNotFoundError, RuntimeError):
            if i == num_retry - 1:
                raise
            print(f"Failed to download assets from HuggingFace dataset. Trying again in {retry_delay}s...")
            time.sleep(retry_delay)
        else:
            break

    return asset_path
