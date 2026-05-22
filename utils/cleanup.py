from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import BACKUP_ROOT, PROJECT_ROOT, repo_relative


@dataclass
class CleanupAction:
    path: str
    exists: bool
    backup_path: str
    backup_already_present: bool
    backed_up: bool
    deleted: bool


def _assert_inside_project(path: Path) -> None:
    try:
        path.absolute().relative_to(PROJECT_ROOT.absolute())
    except ValueError as exc:
        raise ValueError(f"Refusing to clean path outside project: {path}") from exc


def backup_then_delete(path: str | Path, *, backup_root: str | Path = BACKUP_ROOT, execute: bool = False) -> CleanupAction:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    _assert_inside_project(source)
    rel = source.absolute().relative_to(PROJECT_ROOT.absolute())
    target = Path(backup_root).expanduser() / rel
    exists = source.exists()
    backup_present = target.exists()
    backed_up = False
    deleted = False

    if exists and execute:
        if not backup_present:
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target)
            backed_up = True
        if source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
        deleted = True

    return CleanupAction(
        path=repo_relative(source),
        exists=exists,
        backup_path=str(target),
        backup_already_present=backup_present,
        backed_up=backed_up,
        deleted=deleted,
    )


def clean_many(paths: list[str], *, backup_root: str | Path = BACKUP_ROOT, execute: bool = False) -> list[CleanupAction]:
    return [backup_then_delete(path, backup_root=backup_root, execute=execute) for path in paths]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backup project paths before deleting them.")
    parser.add_argument("paths", nargs="+", help="Project-relative files or directories to remove.")
    parser.add_argument("--backup-root", type=Path, default=BACKUP_ROOT)
    parser.add_argument("--execute", action="store_true", help="Actually copy missing backups and delete files.")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    actions = clean_many(args.paths, backup_root=args.backup_root, execute=args.execute)
    payload = [asdict(action) for action in actions]
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for action in actions:
            status = "deleted" if action.deleted else "dry-run"
            if not action.exists:
                status = "missing"
            backup = "already-backed-up" if action.backup_already_present else ("backed-up" if action.backed_up else "needs-backup")
            print(f"{status}: {action.path} -> {action.backup_path} ({backup})")


if __name__ == "__main__":
    main()
