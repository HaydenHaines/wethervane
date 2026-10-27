"""Reproduce retry blockage after an ordinary exception leaves a pid-named temp."""
import tempfile
from pathlib import Path
import core

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "store"
    core.initialize(root)
    core.create_run(root, "r", "explicit_empty")
    core.seal(root, "r")
    original = core.crash

    def injected(selected, point):
        if point == "receipt_before_publish":
            raise OSError("simulated interruption after temp fsync")

    core.crash = injected
    try:
        core.finalize(root, "r")
    except OSError as exc:
        print("first_finalize", type(exc).__name__, str(exc))
    finally:
        core.crash = original
    print("stale_temp", [p.name for p in root.glob("receipt-*")])
    try:
        print("retry", core.finalize(root, "r"))
    except Exception as exc:
        print("retry_error", type(exc).__name__, str(exc))
