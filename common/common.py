"""
Contains code shared by the different modules.
"""

import logging
import json

LOGGER = logging.getLogger(__name__)


def load_and_merge_parameters(parameters, parameter_file_path, ignore_unused_parameters=False):
    """
    # Loads the default parameters and merges it with the parameters provided as input.
    """
    # Load default parameters
    with open(parameter_file_path) as f:
        import os
        print(os.getcwd())
        default_parameters = json.load(f)

    # Merge parameters
    update_dict(parameters, default_parameters, ignore_unused_parameters=ignore_unused_parameters)

    return parameters


def update_dict(parameters, default_parameters, ignore_unused_parameters):
    """
    Update values of a dictionary given a second dictionary containing the to be updated values. The structure of the
    second dict keys/values should match the to be updated dict. New keys or keys at a different place should be
    ignored. The parameters are leading over the default, unless the parameter given is None.
    """
    unknown_parameters = [k for k in parameters.keys() if k not in default_parameters.keys()]
    if unknown_parameters and not ignore_unused_parameters:
        raise ValueError(f"Initialization with unknown parameter(s) {unknown_parameters} which "
                         f"will be ignored.")
    for k, v in default_parameters.items():
        if (k in parameters and isinstance(parameters[k], dict)
                and isinstance(default_parameters[k], dict)):
            update_dict(parameters[k], default_parameters[k], ignore_unused_parameters)
        elif k not in parameters or parameters[k] in (None, {}):
            parameters[k] = default_parameters[k]


def recursive_dict_filling(dct, keys, value):
    """
    Recursively add value to an arbitrary deep dictionary structure based on a list of keys.
    """
    if not keys:
        return value
    else:
        if not keys[0] in dct.keys():
            dct[keys[0]] = dict()
        dct[keys[0]] = recursive_dict_filling(dct[keys[0]], keys[1:], value)
        return dct


def flatten_dict(dct, key_string=""):
    """
    Expects a dictionary where all keys are strings. Returns a list of tuples containing the key vaue pair where the
    keys are merged with a dot in between.
    """
    result = []
    for key in dct:
        updated_key_string = key_string + key + "."
        if isinstance(dct[key], dict):
            result += flatten_dict(dct[key], updated_key_string)
        else:
            result.append((updated_key_string[:-1], dct[key]))
    return result
