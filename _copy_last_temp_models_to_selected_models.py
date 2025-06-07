import os
import shutil
from pathlib import Path


def copy_latest_model_only(source_root: str, target_root: str):
    source_root = Path(source_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for model_family_dir in source_root.iterdir():
        if not model_family_dir.is_dir():
            continue

        # Get subdirectories (assumed to be versioned models)
        subdirs = [d for d in model_family_dir.iterdir() if d.is_dir()]
        if not subdirs:
            continue

        # Find the latest subdir by creation time
        latest_model_dir = max(subdirs, key=lambda d: d.stat().st_ctime)

        destination = target_root / latest_model_dir.name
        shutil.copytree(latest_model_dir, destination, dirs_exist_ok=True)


# Example usage
copy_latest_model_only("temp_models/pipe/reduced_feats", "selected_models/reduced_feats")
