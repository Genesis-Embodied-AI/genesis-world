import fnmatch
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
from huggingface_hub import HfApi, RepoFile, hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError

from genesis.constants import GLTF_FORMATS, MESH_FORMATS, MJCF_FORMAT, URDF_FORMAT, USD_FORMATS
from genesis.recorders.trajectory import TRAJECTORY_FORMAT

REPOSITY_URL = "Genesis-Embodied-AI/Genesis"
DEFAULT_BRANCH_NAME = "main"

HUGGINGFACE_ASSETS_REVISION = "deffbe1685e5ce6aa289efdd541f762f1eb3353f"

MESH_EXTENSIONS = (".mtl", *MESH_FORMATS, *GLTF_FORMATS, *USD_FORMATS)
IMAGE_EXTENSIONS = (".png", ".jpg")

# Get repository "root" path (actually test dir is good enough)
TEST_DIR = os.path.dirname(os.path.dirname(__file__))

# Each line of the lock pins a file of the snapshots repository to its own commit, so that pull requests regenerating
# different snapshots edit different lines and merge without conflict.
SNAPSHOTS_LOCK_PATH = os.path.join(TEST_DIR, "snapshots.lock")


@cache
def get_snapshots_revision():
    with open(SNAPSHOTS_LOCK_PATH, encoding="utf-8") as f:
        return dict(line.split() for line in f)


def update_snapshots_lock(revision):
    """Pin every snapshot that a commit of the snapshots repository changes to that commit.

    The revision is a commit hash of the 'main' branch of the repository, or a unique prefix of one. A snapshot that the
    commit adds gets a new line in the lock, and a snapshot that it deletes loses its line. A snapshot changed by an
    earlier commit keeps its pin, so pass each commit of a series of pushes in turn, from oldest to newest.
    """
    api = HfApi()
    commits_id = [
        commit.commit_id for commit in api.list_repo_commits("Genesis-Intelligence/snapshots", repo_type="dataset")
    ]
    commits_id_match = [commit_id for commit_id in commits_id if commit_id.startswith(revision)]
    if len(commits_id_match) != 1:
        raise ValueError(f"'{revision}' matches no single commit of the 'main' branch of the snapshots repository.")
    commit_id = commits_id_match[0]
    i_commit = commits_id.index(commit_id)

    tree, tree_parent = (
        {
            path_info.path: path_info.blob_id
            for path_info in api.list_repo_tree(
                "Genesis-Intelligence/snapshots", repo_type="dataset", recursive=True, revision=tree_revision
            )
            if isinstance(path_info, RepoFile) and "/__snapshots__/" in path_info.path
        }
        for tree_revision in commits_id[i_commit : i_commit + 2]
    )

    snapshots_revision = dict(get_snapshots_revision())
    for path in tree_parent.keys() - tree.keys():
        print(f"Unpinning deleted snapshot '{path}'.")
        snapshots_revision.pop(path, None)
    for path, blob_id in tree.items():
        if tree_parent.get(path) != blob_id:
            print(f"Pinning snapshot '{path}' to '{commit_id}'.")
            snapshots_revision[path] = commit_id
    with open(SNAPSHOTS_LOCK_PATH, "w", encoding="utf-8") as f:
        f.writelines(sorted(f"{path} {path_revision}\n" for path, path_revision in snapshots_revision.items()))
    get_snapshots_revision.cache_clear()


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

    if repo_name == "snapshots":
        snapshots_revision = get_snapshots_revision()
        assert local_dir is not None
        snapshot_paths = [path for path in snapshots_revision if fnmatch.fnmatchcase(path, pattern)]
        if not snapshot_paths:
            raise ValueError(f"No snapshot in '{SNAPSHOTS_LOCK_PATH}' matches '{pattern}'.")
    elif repo_name != "assets":
        raise ValueError(f"Unsupported repository '{repo_name}'")

    for i in range(num_retry):
        try:
            # Try downloading the assets
            if repo_name == "snapshots":
                for path in snapshot_paths:
                    hf_hub_download(
                        repo_type="dataset",
                        repo_id=f"Genesis-Intelligence/{repo_name}",
                        filename=path,
                        revision=snapshots_revision[path],
                        local_dir=local_dir,
                    )
                asset_path = local_dir
            else:
                asset_path = snapshot_download(
                    repo_type="dataset",
                    repo_id=f"Genesis-Intelligence/{repo_name}",
                    revision=HUGGINGFACE_ASSETS_REVISION,
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
