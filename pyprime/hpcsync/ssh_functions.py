from __future__ import annotations

from .online_pass import get_online_pass

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


# ============================================================
# basic helpers
# ============================================================

def save_dict_to_file(dictionary: dict, file_path: str | Path) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(dictionary, file, indent=4)


def load_dict_from_file(file_path: str | Path) -> dict:
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_numeric(filename: str, out: bool = False) -> int:
    if out:
        print(f'Ermittle Index aus "{filename}"')

    numeric_part = "".join(filter(str.isdigit, filename))
    try:
        numeric_value = int(numeric_part)
        if out:
            print(f"-> {numeric_part}")
        return numeric_value
    except ValueError:
        if out:
            print("no content")
        return 10000


def sort_files(files: Iterable[str]) -> list[str]:
    return sorted(files, key=extract_numeric)


# ============================================================
# ssh / rsync helpers
# ============================================================

def _get_ssh_info(ssh_pass, output: bool = False):
    """
    Expected from get_online_pass:
        hostname, port, username, remote_file_path, local_filesystem
    """
    hostname, port, username, remote_file_path, local_filesystem = get_online_pass(
        ssh_pass, output=output
    )
    return hostname, port, username, remote_file_path, local_filesystem


def _ensure_commands_available() -> None:
    for cmd in ("rsync", "ssh", "scp"):
        if shutil.which(cmd) is None:
            raise RuntimeError(f"Required command '{cmd}' not found.")


def _run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _ssh_base_cmd(port: int) -> list[str]:
    return [
        "ssh",
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
    ]


def _rsync(
    remote_source: str,
    local_target: str | Path,
    hostname: str,
    port: int,
    username: str,
    excludes: list[str] | None = None,
    include_only: list[str] | None = None,
    update_existing: bool = True,
    delete: bool = False,
) -> None:
    _ensure_commands_available()

    local_target = Path(local_target)
    local_target.mkdir(parents=True, exist_ok=True)

    ssh_cmd = (
        f"ssh -p {port} "
        f"-o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null"
    )

    cmd = [
        "rsync",
        "-avz",
        "--info=progress2",
        "-h",
        "-e", ssh_cmd,
    ]

    if not update_existing:
        cmd.append("--ignore-existing")

    if delete:
        cmd.append("--delete")

    if include_only:
        for pattern in include_only:
            cmd += ["--include", pattern]
        cmd += ["--include", "*/", "--exclude", "*"]

    if excludes:
        for pattern in excludes:
            cmd += ["--exclude", pattern]

    remote = f"{username}@{hostname}:{remote_source.rstrip('/')}/"
    local = str(local_target)

    cmd += [remote, local]
    _run_command(cmd)


def _scp_download_file(
    remote_file: str,
    local_file: str | Path,
    hostname: str,
    port: int,
    username: str,
) -> None:
    local_file = Path(local_file)
    local_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "scp",
        "-P", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{username}@{hostname}:{remote_file}",
        str(local_file),
    ]
    _run_command(cmd)


def _ssh_listdir(
    remote_path: str,
    hostname: str,
    port: int,
    username: str,
) -> list[str]:
    cmd = _ssh_base_cmd(port) + [
        f"{username}@{hostname}",
        f"find {shlex.quote(remote_path)} -maxdepth 1 -type f -printf '%f\\n'",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


# ============================================================
# main API
# ============================================================

def ssh_download_sim(
    path,
    ssh_pass,
    ssh_download_fluid: bool = False,
    ssh_update: bool = True,
    bub_fp: bool = False,
):
    hostname, port, username, remote_file_path, local_filesystem = _get_ssh_info(ssh_pass)

    local_filesystem = Path(local_filesystem)
    local_filesystem.mkdir(parents=True, exist_ok=True)

    path_id_dict = local_filesystem / "id_dict.json"
    new_mirror = False

    if path_id_dict.exists():
        id_dict = load_dict_from_file(path_id_dict)

        if path in id_dict:
            print("Simulation data was already downloaded. Checking for updates...")
            lpath = local_filesystem / id_dict[path]
        else:
            if len(id_dict) == 0:
                new_id = 1
            else:
                last_entry = list(id_dict.items())[-1]
                _, last_value = last_entry
                new_id = int(extract_numeric(last_value)) + 1

            llpath = f"sim_results_{new_id}"
            lpath = local_filesystem / llpath
            id_dict[path] = llpath
            print(f"Created a new directory in the local filesystem, ID = {new_id}")
            save_dict_to_file(id_dict, path_id_dict)
            new_mirror = True
    else:
        print("Created the first directory in this local filesystem, ID = 1")
        new_dict = {path: "sim_results_1"}
        save_dict_to_file(new_dict, path_id_dict)
        lpath = local_filesystem / "sim_results_1"
        new_mirror = True

    rpath = str(Path(remote_file_path) / path)
    lpath.mkdir(parents=True, exist_ok=True)

    if ssh_update or new_mirror:
        print("Updating local simulation data...")
        download_directory(
            remote_path=rpath,
            local_path=lpath,
            ssh_pass=ssh_pass,
            bub_fp=bub_fp,
        )
        print(f"SSH data checked. Local directory is {lpath}")

        if ssh_download_fluid:
            download_fluid(rpath, lpath, ssh_pass, I=0)
        else:
            download_fluid(rpath, lpath, ssh_pass, I=-1)

    return str(lpath), rpath


def ssh_get_fluid(i, ssh_pass, lpath, rpath):
    download_fluid(rpath, lpath, ssh_pass, I=i)


def download_directory(remote_path, local_path, ssh_pass, pics: bool = False, bub_fp: bool = False):
    hostname, port, username, _, _ = _get_ssh_info(ssh_pass, output=False)

    exclude_names = [
        "ellipsoid_trn",
        "ellipsoid_rst",
        "fluid",
        "fl_slices",
        "build",
        "postproc",
        "postproc_bubble",
        "preproc",
        "job.sh",
        "Makefile",
        "Report.txt",
        "vis_check",
        "vis_checkT0",
        "fp_00",
    ]

    bubfpnames = [
        "_fp",
        "_MM",
        "_fpf",
        "_tc_",
        "_pri_",
        "_sk_",
        "tau",
        "_fpMu",
        "_fpMu1",
        "_fpMui",
        "_fp_Ct",
        "_fp_unc",
        "_fp_dotc",
        "_fp_un",
        "_fp_ph",
        "_fp_sk",
        "_umove",
        "_fpu",
        "_fpui",
        "_xloci",
        "_xcf",
        "_trn",
        "_rst",
        "_tfe",
        "_tfe_g",
        "marker",
        "_tra",
        "_tra_theta",
        "I1m",
        "errN",
        "JSr",
        "tc",
        "tct",
    ]

    if not bub_fp:
        exclude_names += bubfpnames

    if not pics:
        exclude_names += ["vis_C1", "vis_C2", "vis_A", "vis_S"]

    excludes = [f"*{name}*" for name in exclude_names]

    try:
        _rsync(
            remote_source=remote_path,
            local_target=local_path,
            hostname=hostname,
            port=port,
            username=username,
            excludes=excludes,
            update_existing=True,
            delete=False,
        )

        remote_parent = str(Path(remote_path).parent)
        local_path = Path(local_path)

        extra_files = [
            "input_bubble_data.dat",
            "input_ellipsoid_data.dat",
            "input.dat",
        ]

        for fname in tqdm(extra_files, desc="Extra files", unit="file"):
            local_file = local_path / fname
            if not local_file.exists():
                remote_file = str(Path(remote_parent) / fname)
                try:
                    print(f"Downloading missing extra file: {fname}")
                    _scp_download_file(
                        remote_file=remote_file,
                        local_file=local_file,
                        hostname=hostname,
                        port=port,
                        username=username,
                    )
                except subprocess.CalledProcessError:
                    print(f"Could not download optional file: {fname}")

    except subprocess.CalledProcessError as e:
        print(f"rsync/scp failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def download_fluid(rpath, lpath, ssh_pass, I: int = -1):
    """
    I = -1  -> only x.dat, y.dat, z.dat
    I = 0   -> all fluid files
    I = n   -> only files whose numeric index equals n
    """
    hostname, port, username, _, _ = _get_ssh_info(ssh_pass)

    fluid_remote = str(Path(rpath) / "fluid")
    fluid_local = Path(lpath) / "fluid"
    fluid_local.mkdir(parents=True, exist_ok=True)

    try:
        if I == -1:
            print("Downloading fluid coordinate files only...")
            _rsync(
                remote_source=fluid_remote,
                local_target=fluid_local,
                hostname=hostname,
                port=port,
                username=username,
                include_only=["x.dat", "y.dat", "z.dat"],
                update_existing=True,
                delete=False,
            )
            return

        if I == 0:
            print("Downloading all fluid files...")
            _rsync(
                remote_source=fluid_remote,
                local_target=fluid_local,
                hostname=hostname,
                port=port,
                username=username,
                update_existing=False,
                delete=False,
            )
            return

        remote_files = _ssh_listdir(
            fluid_remote,
            hostname=hostname,
            port=port,
            username=username,
        )

        for fl in tqdm(sort_files(remote_files), desc="Fluid files", unit="file"):
            if fl in {"x.dat", "y.dat", "z.dat"}:
                remote_file = str(Path(fluid_remote) / fl)
                local_file = fluid_local / fl
                if not local_file.exists():
                    print(f"Downloading fluid data - {fl}")
                    _scp_download_file(
                        remote_file=remote_file,
                        local_file=local_file,
                        hostname=hostname,
                        port=port,
                        username=username,
                    )
                continue

            fl_id = extract_numeric(fl)
            if fl_id == I:
                remote_file = str(Path(fluid_remote) / fl)
                local_file = fluid_local / fl
                if not local_file.exists():
                    print(f"Downloading fluid data - {fl}")
                    _scp_download_file(
                        remote_file=remote_file,
                        local_file=local_file,
                        hostname=hostname,
                        port=port,
                        username=username,
                    )

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def ssh_get_files(sim, files):
    hostname, port, username, _, _ = _get_ssh_info(sim.ssh_pass)

    if not isinstance(files, list):
        files = [files]

    try:
        for file in tqdm(files, desc="Downloading files", unit="file"):
            lpath = Path(sim.resultpath) / file
            rpath = str(Path(sim.rpath) / file)
            print(f"Downloading: {Path(file).name}")
            _scp_download_file(
                remote_file=rpath,
                local_file=lpath,
                hostname=hostname,
                port=port,
                username=username,
            )
    except Exception as e:
        print(f"An error occurred: {e}")