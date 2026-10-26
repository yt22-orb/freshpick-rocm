"""Versioned household reference data, separate from image-model predictions."""
import json
from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=1)
def shelf_life_reference():
    return json.loads(files('freshpick').joinpath('shelf_life_reference.json').read_text())


def storage_reference(produce, condition, location):
    return next((row for row in shelf_life_reference()['records']
                 if (row['produce'], row['condition'], row['location']) ==
                 (produce, condition, location)), None)
