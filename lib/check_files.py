"""Functions to download data bundles from Zenodo, if required."""
import os
from shutil import unpack_archive
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Optional, Union

import requests
from tqdm import tqdm

DIRNAME = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PREPARED_DATA_DIR = os.path.join(DIRNAME, "..", "prepared_data")
DEFAULT_MODEL_DIR = os.path.join(DIRNAME, "..", "plate_model")
DEFAULT_SOURCE_DATA_DIR = os.path.join(DIRNAME, "..", "data")

_DOI_URL = "https://doi.org/10.5281/zenodo.8157690"
_ZENODO_URL = "https://zenodo.org/record/8157691"


def check_prepared_data(
    data_dir: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the prepared training and grid data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    if data_dir is None:
        data_dir = DEFAULT_PREPARED_DATA_DIR
    data_dir = os.path.abspath(data_dir)

    if force or (not os.path.isdir(data_dir)):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/prepared_data.zip"

        if verbose:
            print(
                f"Downloading source data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir


def check_source_data(
    data_dir: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the source data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    if data_dir is None:
        data_dir = DEFAULT_SOURCE_DATA_DIR
    data_dir = os.path.abspath(data_dir)

    if force or (not os.path.isdir(data_dir)):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/source_data.zip"

        if verbose:
            print(
                f"Downloading source data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir


def check_plate_model(
    model_dir: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the plate model data bundle.

    Parameters
    ----------
    model_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    model_dir : str
        The location of the downloaded data bundle.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    model_dir = os.path.abspath(model_dir)

    if force or (not os.path.isdir(model_dir)):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/plate_model.zip"

        if verbose:
            print(
                f"Downloading plate model: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=os.path.dirname(model_dir),
            verbose=verbose,
        )
    return model_dir

def _download_extract(
    url: str,
    extract_dir: Union[os.PathLike, str],
    archive_filename: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
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
    # Don't print progress if in a notebook
    # (can result in hundreds of lines of output)
    verbose = verbose and ("get_ipython" not in dir())

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
