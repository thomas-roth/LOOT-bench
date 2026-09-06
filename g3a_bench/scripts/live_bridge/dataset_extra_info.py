"""Sidecar metadata for LeRobot datasets recorded by this bridge.

lerobot 0.6.0 turned ``LeRobotDatasetMetadata.info`` into a closed dataclass
(``DatasetInfo``) with a fixed schema: dict-style ``info[key] = value`` now
raises ``KeyError`` for any field it doesn't declare, and ``write_info``
round-trips only declared fields (``to_dict`` is ``dataclasses.asdict``), so
even a plain ``setattr`` would be silently dropped on save. g3a-bench's own
recording metadata (per-episode RNG seeds, the device they were drawn on,
the operator id, the start-pose noise the episodes were flown under)
doesn't fit that schema, so it lives in a sibling JSON file instead of
``meta/info.json``.
"""

from pathlib import Path

from lerobot.utils.io_utils import load_json, write_json

EXTRA_INFO_FILENAME = "g3a_extra_info.json"


def extra_info_path(root: str | Path) -> Path:
    return Path(root) / "meta" / EXTRA_INFO_FILENAME


def load_extra_info(root: str | Path) -> dict:
    path = extra_info_path(root)
    if not path.exists():
        return {}
    return load_json(path)


def write_extra_info(root: str | Path, info: dict) -> None:
    write_json(info, extra_info_path(root))
