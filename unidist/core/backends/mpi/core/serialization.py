# Copyright (C) 2021-2022 Modin authors
#
# SPDX-License-Identifier: Apache-2.0

"""Serialization interface"""

import importlib
import inspect
import sys
from collections.abc import KeysView

# Serialization libraries
if sys.version_info[1] < 8:  # check the minor Python version
    try:
        import pickle5 as pkl
    except ImportError:
        raise ImportError(
            "Missing dependency 'pickle5'. Use pip or conda to install it."
        ) from None
else:
    import pickle as pkl
import cloudpickle as cpkl

from unidist.core.backends.mpi.core.common import get_logger

logger = get_logger("serialization", "serialization.log")

# Pickle 5 protocol compatible types check
compatible_modules = ("pandas", "numpy")
available_modules = []
for module_name in compatible_modules:
    try:
        available_modules.append(importlib.import_module(module_name))
    except ModuleNotFoundError:
        pass


def is_cpkl_serializable(data):
    """
    Check if the data should be serialized with cloudpickle.

    Parameters
    ----------
    data : object
        Python object

    Returns
    -------
    bool
        ``True` if the data should be serialized with cloudpickle library.
    """
    return (
        inspect.isfunction(data)
        or inspect.isclass(data)
        or inspect.ismethod(data)
        or data.__class__.__module__ != "builtins"
        or isinstance(data, KeysView)
    )


def is_pickle5_serializable(data):
    """
    Check if the data should be serialized with pickle 5 protocol.

    Parameters
    ----------
    data : object
        Python object.

    Returns
    -------
    bool
        ``True`` if the data should be serialized with pickle using protocol 5 (out-of-band data).
    """
    for module in available_modules:
        if module.__name__ == "pandas" and isinstance(
            data, (module.DataFrame, module.Series)
        ):
            return True
        elif module.__name__ == "numpy" and isinstance(data, module.ndarray):
            return True

    return False


def _cpkl_encode(obj):
    """
    Encode with cloudpickle library.

    Parameters
    ----------
    obj : object
        Python object.

    Returns
    -------
    dict
        Dictionary with array of serialized bytes.
    """
    return {"__cloud_custom__": True, "as_bytes": cpkl.dumps(obj)}


def serialize(data):
    """
    Serialize data to a byte array.

    Parameters
    ----------
    data : object
        Data to serialize.

    Notes
    -----
    Uses msgpack, cloudpickle and pickle libraries.
    """
    try:
        if isinstance(data, tuple):
            return tuple(serialize(el) for el in data)
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = serialize(data[i])
            return data
        if type(data) == dict:
            for key in data:
                data[key] = serialize(data[key])
            return data
        if is_pickle5_serializable(data):
            return data
        if is_cpkl_serializable(data):
            return _cpkl_encode(data)
        return data
    except Exception as ex:
        logger.exception(ex)
        raise ex


def deserialize(s_data):
    """
    De-serialize data from a bytearray.

    Parameters
    ----------
    s_data : bytearray
        Data to de-serialize.

    Notes
    -----
    Uses msgpack, cloudpickle and pickle libraries.
    """
    try:
        if isinstance(s_data, tuple):
            return tuple(deserialize(el) for el in s_data)
        if isinstance(s_data, list):
            for i in range(len(s_data)):
                s_data[i] = deserialize(s_data[i])
            return s_data
        if type(s_data) == dict:
            if "__cloud_custom__" in s_data:
                return cpkl.loads(s_data["as_bytes"])
            for key in s_data:
                s_data[key] = deserialize(s_data[key])
            return s_data
        return s_data
    except Exception as ex:
        logger.exception(ex)
        raise ex


class SimpleDataSerializer:
    """
    Class for simple data serialization/de-serialization for MPI communication.

    Notes
    -----
    Uses cloudpickle and pickle libraries as separate APIs.
    """

    def serialize_cloudpickle(self, data):
        """
        Encode with a cloudpickle library.

        Parameters
        ----------
        obj : object
            Python object.

        Returns
        -------
        bytearray
            Array of serialized bytes.
        """
        return cpkl.dumps(data)

    def serialize_pickle(self, data):
        """
        Encode with a pickle library.

        Parameters
        ----------
        obj : object
            Python object.

        Returns
        -------
        bytearray
            Array of serialized bytes.
        """
        return pkl.dumps(data)

    def deserialize_cloudpickle(self, data):
        """
        De-serialization with cloudpickle library.

        Parameters
        ----------
        obj : bytearray
            Python object.

        Returns
        -------
        object
            Original reconstructed object.
        """
        return cpkl.loads(data)

    def deserialize_pickle(self, data):
        """
        De-serialization with pickle library.

        Parameters
        ----------
        obj : bytearray
            Python object.

        Returns
        -------
        object
            Original reconstructed object.
        """
        return pkl.loads(data)
