#!/usr/bin/env python3

import os
import sys

import papermill as pm

# Disable ipykernel warnings
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

ALL_NOTEBOOKS = (
    "00a-extract_training_data",
    "00b-extract_grid_data",
    "00c-extract_training_data_global",
    "01-create_classifiers",
    "02-create_probability_maps",
    "03-create_probability_animations",
    "04-create_erosion_distribution.ipynb",
    "05-create_preservation_maps.ipynb",
    "06-create_preservation_animations.ipynb",
    "07-partial_dependence.ipynb",
)

def run_notebook(
    input_filename: str,
    output_filename=None,
    parameters=None,
):
    if not input_filename.endswith(".ipynb"):
        input_filename += ".ipynb"
    if not os.path.isfile(input_filename):
        raise FileNotFoundError(
            f"Input file not found: {input_filename}"
        )
    if output_filename is None:
        output_filename = input_filename[:-6] + "_output.ipynb"
    print(f"Running notebook: {input_filename}", file=sys.stderr)
    print(f"Output file: {output_filename}", file=sys.stderr)
    pm.execute_notebook(
        input_filename,
        output_filename,
        parameters,
        kernel_name="python3",
        cwd=os.path.dirname(os.path.abspath(input_filename)),
    )


def _main(args):
    if args.list_defaults:
        print(
            "Default notebooks to execute:",
            *[
                f" - {f}.ipynb"
                for f in ALL_NOTEBOOKS
            ],
            sep="\n",
            flush=True,
        )
        return 0
    if len(args.input_filenames) == 0:
        raise ValueError("Must provide at least one input file.")
    for filename in args.input_filenames:
        if args.overwrite:
            output_filename = filename
        else:
            output_filename = None
        run_notebook(filename, output_filename, parameters=None)
    return 0


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Execute notebooks from the command line.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="overwrite input files",
        action="store_true",
        dest="overwrite",
    )
    parser.add_argument(
        "-l",
        "--list-defaults",
        help="list default input files",
        action="store_true",
        dest="list_defaults",
    )
    parser.add_argument(
        metavar="INPUT_FILE",
        help="input notebooks to execute",
        nargs="*",
        dest="input_filenames",
    )
    args = parser.parse_args()
    _main(args)
