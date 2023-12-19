#!/usr/bin/env python3

import os
from multiprocessing import cpu_count
from sys import stderr

# Disable ipykernel warnings
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

CONFIG = {
    "TemplateExporter": {
        "exclude_input_prompt": True,
        "exclude_output_prompt": True,
        "exclude_output": True,
    }
}
ALL_NOTEBOOKS = (
    "00a-extract_training_data",
    "00b-extract_grid_data",
    "00c-extract_training_data_global",
    "01-create_classifiers",
    "01a-create_classifiers_global",
    "01b-cross_validation",
    "02-create_probability_maps",
    "02a-create_probability_maps_global",
    "02b-present_day_probabilities",
    "03-create_probability_animations",
    "03a-create_probability_animations_global",
    "Fig-01-02-probability_snapshots",
    "Fig-03-04-feature_importance",
    "Fig-05-partial_dependence",
    "Fig-06-performance"
)


def convert_notebook(
    filename,
    include_ipython=False,
    overwrite=False,
    config=None,
    quiet=False,
):
    import nbconvert
    import nbformat

    filename = str(filename)
    if filename.endswith(os.extsep + "ipynb"):
        filename = os.extsep.join(
            filename.split(os.extsep)[:-1]
        )

    notebook = nbformat.read(
        filename + os.extsep + "ipynb",
        as_version=nbformat.NO_CONVERT,
    )
    exporter = nbconvert.ScriptExporter(config)
    body, _ = exporter.from_notebook_node(notebook)
    if not include_ipython:
        body = remove_ipython(body)
    write_file(
        filename + os.extsep + "py",
        body,
        force=overwrite,
        quiet=quiet,
    )


def write_file(filename, body, force=False, quiet=False):
    if quiet:
        force = True

    if os.path.exists(filename):
        if not force:
            return
        if not quiet:
            print(f"  Overwriting file: {filename}", file=stderr, flush=True)
    elif not quiet:
        print(f"  Creating file: {filename}", file=stderr, flush=True)
    with open(filename, "w") as f:
        f.write(body)


def remove_ipython(s):
    return "\n".join(
        [
            line for line in str(s).split("\n")
            if not line.lstrip().startswith("get_ipython().run_line_magic")
        ]
    )


def _convert(args=None):
    if args is None:
        filenames = ALL_NOTEBOOKS
        overwrite = False
        jobs = min((cpu_count() - 1, 8))
        quiet = False
    else:
        overwrite = args.overwrite
        jobs = args.jobs
        quiet = args.quiet
        if len(args.filenames) > 0:
            filenames = args.filenames
        else:
            filenames = ALL_NOTEBOOKS

    tmp = []
    for filename in filenames:
        with_ext = filename + os.extsep + "py"
        if os.path.exists(with_ext) and not overwrite:
            while True:
                message = f"  {with_ext} already exists; overwrite? (Y/n)  "
                response = input(message).lower()
                if response in {"y", "yes"}:
                    tmp.append(filename)
                    break
                elif response in {"n", "no"}:
                    break
        else:
            tmp.append(filename)
    filenames = tmp
    del tmp

    if jobs == 1:
        for filename in list(filenames):
            convert_notebook(
                filename=filename,
                include_ipython=False,
                config=CONFIG,
                overwrite=overwrite,
                quiet=quiet,
            )
    else:
        import joblib

        with joblib.Parallel(jobs) as parallel:
            parallel(
                joblib.delayed(convert_notebook)(
                    filename=filename,
                    include_ipython=False,
                    config=CONFIG,
                    overwrite=overwrite,
                    quiet=quiet,
                )
                for filename in list(filenames)
            )


def _clean(args):
    if len(args.filenames) > 0:
        filenames = args.filenames
    else:
        filenames = ALL_NOTEBOOKS
    for filename in filenames:
        if not filename.endswith(os.extsep + "py"):
            filename += os.extsep + "py"
        if not os.path.exists(filename):
            if not args.quiet:
                print(f"  File not found: {filename}", file=stderr)
            continue
        if not os.path.isfile(filename):
            if not args.quiet:
                print(f"  Is a directory: {filename}", file=stderr)
            continue
        os.unlink(filename)
        if not args.quiet:
            print(f"  Deleted file: {filename}", file=stderr)


if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    if len(sys.argv) == 1:
        _convert()
        sys.exit(0)

    parser = ArgumentParser(
        description="Convert Jupyter notebooks to Python scripts.",
    )
    subparsers = parser.add_subparsers()

    convert_parser = subparsers.add_parser(
        "convert",
        description="Convert notebooks to scripts.",
    )
    convert_parser.add_argument(
        "-f",
        "--force",
        help="overwrite any existing .py files",
        action="store_true",
        dest="overwrite",
    )
    convert_parser.add_argument(
        "-j",
        "--jobs",
        help="number of threads to use (default: min(number of CPUs - 1, 8))",
        type=int,
        default=min((cpu_count() - 1, 8)),
        dest="jobs",
    )
    convert_parser.add_argument(
        "-q",
        "--quiet",
        help="suppress logging output",
        action="store_true",
        dest="quiet",
    )
    convert_parser.add_argument(
        metavar="FILE",
        nargs="*",
        help=(
            "notebooks to convert (if not specified, "
            "will convert all notebooks)"
        ),
        dest="filenames",
    )
    convert_parser.set_defaults(func=_convert)

    clean_parser = subparsers.add_parser(
        "clean",
        description="Clean up converted scripts.",
    )
    clean_parser.add_argument(
        "-q",
        "--quiet",
        help="suppress logging output",
        action="store_true",
        dest="quiet",
    )
    clean_parser.add_argument(
        metavar="FILE",
        nargs="*",
        help=(
            "scripts to remove (if not specified, "
            "will remove all scripts)"
        ),
        dest="filenames",
    )
    clean_parser.set_defaults(func=_clean)

    args = parser.parse_args()
    args.func(args)
