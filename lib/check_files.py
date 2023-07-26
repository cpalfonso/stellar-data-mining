import os
from shutil import unpack_archive
from sys import stderr
from tempfile import TemporaryDirectory

import requests
from tqdm import tqdm

DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PREPARED_DATA_DIR = os.path.join(DIRNAME, "..", "prepared_data")
DEFAULT_MODEL_DIR = os.path.join(DIRNAME, "..", "plate_model")
DEFAULT_SOURCE_DATA_DIR = os.path.join(DIRNAME, "..", "data")

_ZENODO_URL = "https://zenodo.org/record/8157691"
_PREPARED_DATA_URL = f"{_ZENODO_URL}/files/prepared_data.zip"
_MODEL_URL = f"{_ZENODO_URL}/files/plate_model.zip"
_SOURCE_DATA_URL = f"{_ZENODO_URL}/files/source_data.zip"


def check_prepared_data(data_dir=None, verbose=False, force=False):
    if data_dir is None:
        data_dir = DEFAULT_PREPARED_DATA_DIR
    data_dir = os.path.abspath(data_dir)

    if force or (not os.path.isdir(data_dir)):
        if verbose:
            print(
                f"Downloading source data: {_PREPARED_DATA_URL}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=_PREPARED_DATA_URL,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir


def check_source_data(data_dir=None, verbose=False, force=False):
    if data_dir is None:
        data_dir = DEFAULT_SOURCE_DATA_DIR
    data_dir = os.path.abspath(data_dir)

    if force or (not os.path.isdir(data_dir)):
        if verbose:
            print(
                f"Downloading source data: {_SOURCE_DATA_URL}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=_SOURCE_DATA_URL,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir


def check_plate_model(model_dir=None, verbose=False, force=False):
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir = os.path.abspath(model_dir)

    if force or (not os.path.isdir(model_dir)):
        if verbose:
            print(
                f"Downloading plate model: {_MODEL_URL}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=_MODEL_URL,
            extract_dir=os.path.dirname(model_dir),
            verbose=verbose,
        )
    return model_dir


def _download_extract(
    url,
    extract_dir,
    archive_filename=None,
    verbose=False,
):
    if archive_filename is None:
        tempdir = TemporaryDirectory()
        download_dir = tempdir.name
        basename = "data.zip"
        archive_filename = os.path.join(
            download_dir,
            basename,
        )
    else:
        tempdir = None
        download_dir = os.path.dirname(os.path.abspath(archive_filename))
        basename = os.path.basename(archive_filename)
    archive_filename = _fetch_data(
        url=url,
        download_dir=download_dir,
        filename=basename,
        verbose=verbose,
    )
    unpack_archive(archive_filename, extract_dir=extract_dir)
    if tempdir is not None:
        tempdir.cleanup()


def _fetch_data(url, download_dir, filename=None, verbose=False):
    if filename is None:
        filename = "data.zip"
    filename = os.path.join(download_dir, filename)

    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        it = response.iter_content(chunk_size=1024)
        if verbose:
            it = tqdm(it, unit="kB")
        for data in it:
            f.write(data)
    return filename


if __name__ == "__main__":
    check_prepared_data(verbose=True)
    check_source_data(verbose=True)
    check_plate_model(verbose=True)
