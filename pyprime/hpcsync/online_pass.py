from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import os


# --------------------------------------------------
# Find config file automatically
# --------------------------------------------------
def _find_project_root(start: Path) -> Path:
    """
    Search upwards for remote_config.json or remote_config.example.json.
    """
    for parent in [start] + list(start.parents):
        if (parent / "remote_config.json").exists():
            return parent
        if (parent / "remote_config.example.json").exists():
            return parent
    raise FileNotFoundError("No remote_config.json or remote_config.example.json found in parent directories.")


def _get_config_path(config_path: str | Path | None = None) -> Path:
    """
    Resolve config path.
    """
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    root = _find_project_root(Path(__file__).resolve())

    config = root / "remote_config.json"
    example = root / "remote_config.example.json"

    if config.exists():
        return config
    if example.exists():
        print("Warning: using remote_config.example.json. Create remote_config.json for your real setup.")
        return example

    raise FileNotFoundError("No config file found.")


# --------------------------------------------------
# Load config
# --------------------------------------------------
def _load_remote_config(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = _get_config_path(config_path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Top-level JSON must be a dictionary: {path}")

    return data


def _validate_entry(key: str, entry: dict[str, Any]) -> None:
    required = (
        "hostname",
        "port",
        "username",
        "remote_file_path",
        "local_filesystem",
    )

    missing = [field for field in required if field not in entry]
    if missing:
        raise KeyError(f"Entry '{key}' is missing required fields: {missing}")


# --------------------------------------------------
# Public API
# --------------------------------------------------
def list_online_passes(config_path: str | Path | None = None) -> list[str]:
    data = _load_remote_config(config_path)
    return sorted(data.keys())


def get_online_pass(
    key: str,
    output: bool = True,
    config_path: str | Path | None = None,
) -> tuple[str, int, str, str, str]:
    """
    Returns:
        (hostname, port, username, remote_file_path, local_filesystem)
    """
    data = _load_remote_config(config_path)

    if key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Online pass '{key}' not found. Available keys: {available}")

    entry = data[key]
    _validate_entry(key, entry)

    hostname = str(entry["hostname"])
    sshhost = str(entry["sshhost"])
    port = int(entry["port"])
    username = str(entry["username"])
    remote_file_path = str(entry["remote_file_path"])
    local_filesystem = str(entry["local_filesystem"])

    if output:
        print(f'SSH config valid: "{key}"')

    return (
        hostname,
        sshhost,
        port,
        username,
        remote_file_path,
        local_filesystem,
    )


def get_online_pass_dict(
    key: str,
    output: bool = True,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    data = _load_remote_config(config_path)

    if key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(f"Online pass '{key}' not found. Available keys: {available}")

    entry = data[key]
    _validate_entry(key, entry)

    result = {
        "hostname": str(entry["hostname"]),
        "sshhost": str(entry["sshhost"]),
        "port": int(entry["port"]),
        "username": str(entry["username"]),
        "remote_file_path": str(entry["remote_file_path"]),
        "local_filesystem": str(entry["local_filesystem"]),
    }

    if output:
        print(f'SSH config valid: "{key}"')

    return result


# --------------------------------------------------
# Debug
# --------------------------------------------------
if __name__ == "__main__":
    try:
        print("Config file:", _get_config_path())
        print("Available keys:", list_online_passes())
    except Exception as e:
        print(f"Error: {e}")