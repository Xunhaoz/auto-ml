import json

import numpy as np
import pandas as pd


def is_array_like(obj):
    return isinstance(obj, (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index))


def is_number_like(obj):
    if isinstance(obj, int):
        return False
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        return False


def recursive_dict_iter(d):
    """
    將 dict 中不可以 json 化的物件轉成可以 json 化的物件
    如： np.ndarray, pd.Series, np.int64, np.float64
    """
    new_dict = {}
    for key, value in d.items():
        new_key = str(key) if isinstance(key, int) else key
        if isinstance(value, dict):
            new_dict[new_key] = recursive_dict_iter(value)
        elif is_array_like(value):
            new_array = list(value)
            for k, v in enumerate(new_array):
                if v is is_number_like(v):
                    new_array[k] = round(float(v), 4)
            new_dict[new_key] = new_array
        elif is_number_like(value):
            new_dict[new_key] = round(float(value), 4)
        else:
            new_dict[new_key] = value
    return new_dict


def save_dict_2_json(d, path):
    d = recursive_dict_iter(d)
    with open(path, 'w') as f:
        json.dump(d, f)