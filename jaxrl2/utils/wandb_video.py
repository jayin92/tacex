import os
import shutil
from pathlib import Path
from typing import Optional, Set

import wandb


def log_new_videos(
        video_dir: Optional[str],
        seen_paths: Set[str],
        key_prefix: str,
        step: int,
        enabled: bool = True,
) -> None:
    """Log newly created videos within a directory to Weights & Biases.

    Args:
        video_dir: Directory that may contain video files produced by RecordVideo.
        seen_paths: Mutable set used to track which video paths have already been logged.
        key_prefix: Key under which the videos will be logged, e.g. "train_videos".
        step: Global training step associated with the videos.
        enabled: Flag indicating whether wandb logging should be performed.
    """
    if not enabled:
        return

    if not wandb.run:
        return

    if not video_dir:
        return

    dir_path = Path(video_dir)
    if not dir_path.is_dir():
        return

    def _is_wandb_generated(path: Path) -> bool:
        for parent in path.parents:
            if parent == dir_path:
                break
            if parent.name.startswith('wandb_'):
                return True
        return False

    video_paths = sorted(
        (
            path for path in dir_path.rglob('*.mp4')
            if os.path.isfile(path) and not _is_wandb_generated(path)
        ),
        key=lambda p: p.stat().st_mtime,
    )
    new_videos = []
    split_name = key_prefix.split('_', 1)[0]
    split_dir = dir_path / f'wandb_{split_name}'
    split_dir.mkdir(parents=True, exist_ok=True)

    for path in video_paths:
        resolved_path = str(path.resolve())
        if resolved_path in seen_paths:
            continue

        new_file_name = f'{split_name}_step{step:09d}_{path.name}'
        target_path = split_dir / new_file_name
        if not target_path.exists():
            try:
                os.link(resolved_path, target_path)
            except OSError:
                shutil.copy2(resolved_path, target_path)

        new_videos.append(
            wandb.Video(
                str(target_path),
                format='mp4',
                caption=target_path.stem,
            )
        )
        seen_paths.add(resolved_path)

    if new_videos:
        wandb.log({key_prefix: new_videos}, step=step)
