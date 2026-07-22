"""Small cross-platform helpers for locking an open file."""

from __future__ import annotations

import os
from typing import IO, Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def acquire_exclusive_file_lock(file: IO[Any]) -> None:
    """Block until an exclusive, process-wide lock is held for ``file``."""
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def release_file_lock(file: IO[Any]) -> None:
    """Release a lock previously acquired by :func:`acquire_exclusive_file_lock`."""
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
