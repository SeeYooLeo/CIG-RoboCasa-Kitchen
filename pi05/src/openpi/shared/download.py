import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse
import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"
logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    cache_dir = (
        pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR))
        .expanduser()
        .resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()
    cache_dir = get_cache_dir()
    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path
    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            _ensure_permissions(lock_path)
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()
            logger.info(f"Downloading {url} to {local_path}")
            scratch_path = local_path.with_suffix(".partial")
            _download_fsspec(url, scratch_path, **kwargs)
            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)
    except PermissionError as e:
        msg = f"Local file permission error was encountered while downloading {url}. Please try again after removing the cached data using: `rm -rf {local_path}*`"
        raise PermissionError(msg) from e
    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    if is_dir := (
        info["type"] == "directory"
        or (info["size"] == 0 and info["name"].endswith("/"))
    ):
        total_size = fs.du(url)
    else:
        total_size = info["size"]
    with tqdm.tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(
                (
                    f.stat().st_size
                    for f in [*local_path.rglob("*"), local_path]
                    if f.is_file()
                )
            )
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _set_permission(path: pathlib.Path, target_permission: int):
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        file_rw = (
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
            | stat.S_IWOTH
        )
        if file_path.stat().st_mode & 64:
            _set_permission(
                file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)
        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    return False
