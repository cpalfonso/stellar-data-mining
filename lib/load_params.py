import os
from copy import deepcopy

from ruamel.yaml import YAML

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "notebook_parameters.yml",
)


def get_params(
    filename=_DEFAULT_CONFIG,
    notebook=None,
):
    yaml = YAML()
    with open(filename, "r") as f:
        data = yaml.load(f)

    defaults = data["defaults"]

    # Parameters for all notebooks
    ## Start with defaults
    params = deepcopy(defaults.get("all_notebooks", dict()))
    ## Override with values for all notebooks
    params.update(data.get("all_notebooks", dict()))
    if notebook in {None, "all", "all_notebooks"}:
        return params

    # Parameters for specific notebooks
    notebook = str(notebook).lower()
    if not notebook.startswith("notebook_"):
        notebook = "notebook_" + notebook
    ## Add defaults for specific notebook
    for key, value in defaults.get(notebook, dict()).items():
        if key not in params.keys():
            params[key] = value
    ## Add specified values for notebook
    for key, value in data.get(notebook, dict()).items():
        if (key not in params.keys()) or (value is not None):
            params[key] = value
    return params
