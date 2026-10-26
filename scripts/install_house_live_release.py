"""Atomically install a checked sibling directory without replacing any entry.

Linux renameat2(RENAME_NOREPLACE) is mandatory. There is deliberately no rename
or copy fallback on unsupported platforms, kernels, or filesystems.
System call contract: https://man7.org/linux/man-pages/man2/rename.2.html
"""
import argparse
import ctypes
import os
import stat
import sys


def install(parent, source, destination):
    if sys.platform != "linux":
        raise RuntimeError("Live House installation requires Linux renameat2(RENAME_NOREPLACE).")
    if not os.path.isabs(parent) or os.path.realpath(parent) != parent:
        raise ValueError("Live House installation parent must be an absolute canonical directory.")
    if any(not name or name in (".", "..") or "/" in name or "\0" in name
           for name in (source, destination)) or source == destination:
        raise ValueError("Live House installation requires distinct sibling basenames.")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("This platform does not expose renameat2; installation is unavailable.")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Inspect the opened directory itself: a changed ancestor symlink must
        # not silently redirect installation to a different canonical parent.
        if os.readlink(f"/proc/self/fd/{parent_fd}") != parent:
            raise ValueError("Live House installation parent changed or traverses symlinks.")
        if not stat.S_ISDIR(os.stat(source, dir_fd=parent_fd, follow_symlinks=False).st_mode):
            raise ValueError("Live House installation source must be a directory, not a symlink.")
        if renameat2(parent_fd, os.fsencode(source), parent_fd, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), destination)
    finally:
        os.close(parent_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    install(args.parent, args.source, args.destination)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Live House release installation failed: {error}", file=sys.stderr)
        sys.exit(1)
