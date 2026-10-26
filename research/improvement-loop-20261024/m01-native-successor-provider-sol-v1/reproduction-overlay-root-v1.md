# Recreating the local schema overlay

The frozen test used the established calibration environment's numerical stack and a small, read-only overlay of schema packages already installed in the integration worktree. The overlay links are local runtime artifacts, not committed source files or a production environment. If that directory is absent on this same machine, recreate it from the lab root with:

```sh
python3 - <<'PY'
from pathlib import Path

source = Path('/home/hayden/projects/wethervane-redesign-20261006/integration/.venv/lib/python3.12/site-packages')
overlay = Path('research/improvement-loop-20261024/m01-native-successor-provider-sol-v1/schema-overlay')
names = [
    'pydantic', 'pydantic-2.12.5.dist-info',
    'pydantic_core', 'pydantic_core-2.41.5.dist-info',
    'annotated_types', 'annotated_types-0.7.0.dist-info',
    'typing_inspection', 'typing_inspection-0.4.2.dist-info',
]
overlay.mkdir(parents=True, exist_ok=True)
for name in names:
    target, link = source / name, overlay / name
    if not target.exists():
        raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError(f'unexpected overlay target: {link}')
    elif link.exists():
        raise ValueError(f'overlay path already occupied: {link}')
    else:
        link.symlink_to(target, target_is_directory=True)
PY
```

This does not install, alter or validate the underlying packages. The original run records actual imported package versions and paths in `result.json`; a different installation or machine needs a separately tested environment and must not be assumed numerically equivalent. The frozen `reference-mixture.npz` is required for the reference check and remains source-bound. The author’s full reproducibility commands and the unresolved full-gradient failure are in `FINDINGS.md`.
