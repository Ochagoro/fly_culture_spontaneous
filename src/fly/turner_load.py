"""Load Turner, Mann & Clandinin (2021) region time series (Ito atlas) safely.

The shared files are pandas pickles. A restricted unpickler only allows the
numpy/pandas classes that these files need, so no other code can run.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("pandas.core.frame", "DataFrame"), ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.numeric", "Int64Index"), ("pandas.core.indexes.numeric", "Float64Index"),
    ("pandas.core.indexes.numeric", "UInt64Index"),
    ("pandas.core.internals.managers", "BlockManager"), ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.internals.blocks", "new_block"), ("builtins", "slice"),
    ("pandas.core.internals.managers", "SingleBlockManager"), ("pandas.core.series", "Series"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        if module == "pandas.core.indexes.numeric":
            return pd.Index
        if module.startswith("numpy.core"):
            module = module.replace("numpy.core", "numpy._core")
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()


if __name__ == "__main__":
    obj = load(sys.argv[1])
    print(type(obj))
    if isinstance(obj, pd.DataFrame):
        print(obj.shape); print(obj.iloc[:5, :5]); print(list(obj.index)[:80])
