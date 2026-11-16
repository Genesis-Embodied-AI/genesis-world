import fnmatch
import hashlib
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
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.hf_api import RepoFile
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

SNAPSHOTS_REPO_ID = "Genesis-Intelligence/snapshots"

# Revision of a lock line whose local file '--snapshot-update' regenerated, until '--snapshot-sync' pins it
SNAPSHOTS_PENDING_REVISION = "pending"


@cache
def get_snapshots_revision():
    with open(SNAPSHOTS_LOCK_PATH, encoding="utf-8") as f:
        return dict(line.split() for line in f)


def write_snapshots_lock(snapshots_revision):
    with open(SNAPSHOTS_LOCK_PATH, "w", encoding="utf-8") as f:
        f.writelines(sorted(f"{path} {path_revision}\n" for path, path_revision in snapshots_revision.items()))
    get_snapshots_revision.cache_clear()


def get_snapshots_digest(pattern):
    """Return the SHA-256 digest of every local snapshot file matching a pattern relative to the tests root, by path."""
    return {
        path.relative_to(TEST_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(TEST_DIR).glob(pattern)
    }


def mark_snapshots_pending(snapshots_digest):
    """Mark the snapshots that a regeneration changed as pending in the lock, until '--snapshot-sync' publishes them.

    The argument maps the pattern of each regenerated test to the digests of its files before the regeneration, as
    'get_snapshots_digest' returns them. A file whose bytes differ, or that the test writes for the first time, becomes
    pending. A file that the test no longer writes loses its line.
    """
    snapshots_revision = dict(get_snapshots_revision())
    for pattern, pattern_digest in snapshots_digest.items():
        pattern_digest_new = get_snapshots_digest(pattern)
        for path in pattern_digest.keys() | pattern_digest_new.keys():
            if pattern_digest.get(path) != pattern_digest_new.get(path):
                if path in pattern_digest_new:
                    print(f"Marking regenerated snapshot '{path}' as pending.")
                    snapshots_revision[path] = SNAPSHOTS_PENDING_REVISION
                else:
                    print(f"Removing snapshot '{path}', which its test no longer writes.")
                    del snapshots_revision[path]
    write_snapshots_lock(snapshots_revision)


def get_snapshots_pull_requests():
    """Return the pull requests of the snapshots repository that belong to the current branch, newest first.

    They are the Hugging Face pull requests whose title names the genesis-world pull request of the current git branch
    ('genesis-world#<number>'), which the GitHub CLI (gh) looks up, so a title may carry any other text. The title of a
    new pull request and the address of the genesis-world pull request are returned first.
    """
    try:
        pr_number, pr_url = subprocess.check_output(
            ["gh", "pr", "view", "--json", "number,url", "--jq", ".number, .url"], cwd=TEST_DIR, encoding="utf-8"
        ).split()
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "Publishing snapshots requires a genesis-world pull request for the current branch, found by the GitHub CLI "
            "'gh'. Open one first, as a draft if needed."
        ) from e
    title = f"Update snapshots for genesis-world#{pr_number}"
    discussions = [
        discussion
        for discussion in HfApi().get_repo_discussions(
            repo_id=SNAPSHOTS_REPO_ID, discussion_type="pull_request", repo_type="dataset"
        )
        if re.search(rf"\bgenesis-world#{pr_number}\b", discussion.title)
    ]
    return title, pr_url, discussions


def sync_snapshots():
    """Publish the pending snapshots of the lock to the snapshots repository, and pin those that are merged.

    A pending snapshot whose local file matches its content at the merge commit of a merged pull request returned by
    'get_snapshots_pull_requests' is pinned to that commit, the newest merge first. The other pending snapshots are
    uploaded in one commit to the open pull request of the branch, or to a new draft pull request, and stay pending
    until it is merged.
    """
    snapshots_revision = dict(get_snapshots_revision())
    snapshot_paths = [path for path, revision in snapshots_revision.items() if revision == SNAPSHOTS_PENDING_REVISION]
    if not snapshot_paths:
        return

    title, pr_url, discussions = get_snapshots_pull_requests()
    api = HfApi()
    for discussion in discussions:
        if discussion.status != "merged" or not snapshot_paths:
            continue
        commit_id = api.get_discussion_details(
            repo_id=SNAPSHOTS_REPO_ID, discussion_num=discussion.num, repo_type="dataset"
        ).merge_commit_oid
        for repo_file in api.get_paths_info(
            repo_id=SNAPSHOTS_REPO_ID, paths=snapshot_paths, revision=commit_id, repo_type="dataset"
        ):
            data = Path(TEST_DIR, repo_file.path).read_bytes()
            # The repository identifies a file stored by Git LFS by its SHA-256, and any other one by its Git blob hash
            if repo_file.lfs is not None:
                is_merged = repo_file.lfs.sha256 == hashlib.sha256(data).hexdigest()
            else:
                is_merged = repo_file.blob_id == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if is_merged:
                print(f"Pinning snapshot '{repo_file.path}' to '{commit_id}'.")
                snapshots_revision[repo_file.path] = commit_id
        snapshot_paths = [path for path in snapshot_paths if snapshots_revision[path] == SNAPSHOTS_PENDING_REVISION]
    write_snapshots_lock(snapshots_revision)

    if snapshot_paths:
        discussion = next((discussion for discussion in discussions if discussion.status in ("draft", "open")), None)
        if discussion is None:
            discussion = api.create_pull_request(
                repo_id=SNAPSHOTS_REPO_ID, title=title, description=pr_url, repo_type="dataset"
            )
        api.create_commit(
            repo_id=SNAPSHOTS_REPO_ID,
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=os.path.join(TEST_DIR, path))
                for path in snapshot_paths
            ],
            commit_message=f"Regenerate {len(snapshot_paths)} snapshot(s).",
            repo_type="dataset",
            revision=discussion.git_reference,
        )
        print(
            f"Synced {len(snapshot_paths)} pending snapshot(s) with {discussion.url}. Run '--snapshot-sync' again once "
            "it is merged."
        )


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
        # A pending snapshot exists only as the local file that '--snapshot-update' wrote
        for path in snapshot_paths:
            if snapshots_revision[path] == SNAPSHOTS_PENDING_REVISION and not Path(local_dir, path).is_file():
                raise ValueError(
                    f"Snapshot '{path}' is pending in '{SNAPSHOTS_LOCK_PATH}'. Run '--snapshot-sync' where it was "
                    "regenerated to publish it, then again once its pull request is merged to pin it."
                )
        snapshot_paths = [path for path in snapshot_paths if snapshots_revision[path] != SNAPSHOTS_PENDING_REVISION]
        # A local file the lock no longer lists would count as an unused snapshot and fail the session
        for path in Path(local_dir).glob(pattern):
            if path.relative_to(local_dir).as_posix() not in snapshots_revision:
                path.unlink()
    elif repo_name != "assets":
        raise ValueError(f"Unsupported repository '{repo_name}'")

    for i in range(num_retry):
        try:
            # Try downloading the assets
            if repo_name == "snapshots":
                for path in snapshot_paths:
                    hf_hub_download(
                        repo_type="dataset",
                        repo_id=SNAPSHOTS_REPO_ID,
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
