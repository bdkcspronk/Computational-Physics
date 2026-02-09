import os
import sys


DEFAULT_OPEN_DIR = "ising_sim"
DEFAULT_SNAPSHOT_FILE = "snapshots.npz"


def get_runtime_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resolve_open_dir(open_dir=DEFAULT_OPEN_DIR):
    if os.path.isabs(open_dir):
        return open_dir
    return os.path.join(get_runtime_base_dir(), open_dir)


def ensure_open_dir(open_dir=DEFAULT_OPEN_DIR):
    resolved_dir = resolve_open_dir(open_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    return resolved_dir


def resolve_snapshot_file(open_dir=DEFAULT_OPEN_DIR, file_name=DEFAULT_SNAPSHOT_FILE):
    return os.path.join(resolve_open_dir(open_dir), file_name)
