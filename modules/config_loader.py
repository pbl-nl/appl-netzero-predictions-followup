import json
import os
from copy import deepcopy


MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULES_DIR)
CONFIG_PATH = os.path.join(MODULES_DIR, "config.json")
PATH_KEYS = (
    "input_path",
    "output_path",
    "merged_csv",
    "revised_nzt_workbook",
    "reproduce2_output_path",
)


def resolve_project_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def resolve_config_paths(config):
    resolved = deepcopy(config)
    for key in PATH_KEYS:
        if key in resolved:
            resolved[key] = resolve_project_path(resolved[key])
    return resolved


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return resolve_config_paths(config)
