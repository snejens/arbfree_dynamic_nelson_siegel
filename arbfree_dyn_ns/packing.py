import functools
import operator

import numpy as np
import pandas as pd

from .utils import spectral_radius, spectral_radius_numba

SPECTRAL_RADIUS_CONSTANT = 0


def pack_tightly(objs, spd=tuple(), diag=tuple(), unit_spectrum=tuple(), lower_triangle=tuple()):
    return np.concatenate([flatten(obj, spd=i in spd, diag=i in diag, unit_spectrum=i in unit_spectrum,
                                   lower_triangle=i in lower_triangle)
                           for i, obj in enumerate(objs)])


def unpack(arr, shapes, spd=tuple(), diag=tuple(), unit_spectrum=tuple(), lower_triangle=tuple(), scale=dict(),
           to_pandas=False):
    ret_val = []
    for i, shape in enumerate(shapes):
        i_spd, i_diag, i_unit_spectrum, i_scale, i_lower_triangle = (
        i in spd, i in diag, i in unit_spectrum, scale.get(i),
        i in lower_triangle)
        if i_diag:
            assert shape[0] == shape[1]
            n = shape[0]
        elif i_spd or i_lower_triangle:
            assert shape[0] == shape[1]
            n = shape[0] * (shape[0] + 1) // 2
        elif i_scale is not None:
            n = 1
        else:
            n = functools.reduce(operator.mul, shape)
        arr_beg, arr = arr[:n], arr[n:]
        to_be_appended = unflatten(arr_beg, shape, spd=i_spd, diag=i_diag, unit_spectrum=i_unit_spectrum,
                                   scale=i_scale, lower_triangle=i_lower_triangle)
        if to_pandas:
            to_be_appended = (pd.Series if len(shape) == 1 else pd.DataFrame)(to_be_appended)
        ret_val.append(to_be_appended)
    return ret_val


def flatten(obj, spd=False, diag=False, unit_spectrum=False, lower_triangle=False):
    if hasattr(obj, "to_numpy"):
        obj = obj.to_numpy()

    if unit_spectrum:
        obj = obj.copy()
        obj /= 1 - spectral_radius(obj) - SPECTRAL_RADIUS_CONSTANT

    if obj.ndim != 2 and spd:
        raise ValueError("spd only allowed to be True if obj is 2-dimensional")
    if diag:
        out = obj[np.diag_indices_from(obj)]
        if spd:
            out = np.log(out)
        return out
    elif spd or lower_triangle:
        assert obj.shape[0] == obj.shape[1]
        if spd:
            obj = np.linalg.cholesky(obj)
        return obj[np.tril_indices(obj.shape[0])]
    else:
        return obj.reshape(-1)


def unflatten(obj, shape, spd=False, diag=False, unit_spectrum=False, scale=None, lower_triangle=False):
    if scale is not None:
        out = obj.item() * scale
    elif diag:
        out = np.zeros((obj.shape[0], obj.shape[0]), dtype=obj.dtype)
        out[np.diag_indices_from(out)] = np.exp(obj) if spd else obj
    elif spd or lower_triangle:
        if len(shape) != 2:
            raise ValueError(
                "spd/lower_triangle only allowed to be True if len(shape) == 2, but (obj, shape) = (%s, %s)" % (obj,
                                                                                                                shape))
        assert shape[0] == shape[1]
        n = shape[0]
        R, C = np.tril_indices(n)
        out = np.zeros((n, n), dtype=obj.dtype)
        out[R, C] = obj
        if spd:
            out = out @ out.T
    else:
        out = obj.copy().reshape(shape)

    if unit_spectrum:
        out /= 1 + spectral_radius_numba(out) + SPECTRAL_RADIUS_CONSTANT

    return out
