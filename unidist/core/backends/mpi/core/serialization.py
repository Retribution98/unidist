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
import msgpack
import gc  # msgpack optimization
import psutil
import os
import unidist.core.backends.mpi.core.common as common

logger = common.get_logger('serialization', 'serialization.log')

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
    is_serializable = False
    for module in available_modules:
        if module.__name__ == "pandas":
            is_serializable = isinstance(data, (module.DataFrame, module.Series))
        elif module.__name__ == "numpy":
            is_serializable = isinstance(data, module.ndarray)
    return is_serializable


class ComplexDataSerializer:
    """
    Class for data serialization/de-serialization for MPI comminication.

    Parameters
    ----------
    buffers : list, default: None
        A list of ``PickleBuffer`` objects for data decoding.
    len_buffers : list, default: None
        A list of buffer sizes for data decoding.

    Notes
    -----
    Uses a combination of msgpack, cloudpickle and pickle libraries
    """

    def __init__(self, buffers=None, len_buffers=None):
        self.buffers = buffers if buffers else []
        self.len_buffers = len_buffers if len_buffers else []
        self._callback_counter = 0

    def _buffer_callback(self, pickle_buffer):
        """
        Callback for pickle protocol 5 out-of-band data buffers collection.

        Parameters
        ----------
        pickle_buffer: pickle.PickleBuffer
            Pickle library buffer wrapper.
        """
        self.buffers.append(pickle_buffer)
        self._callback_counter += 1
        return False

    def _dataframe_encode(self, frame):
        """
        Encode with pickle library using protocol 5.

        Parameters
        ----------
        data : object
            Pickle 5 serializable object (e.g. pandas DataFrame or NumPy array).

        Returns
        -------
        dict
            Dictionary with array of serialized bytes.
        """
        s_frame = pkl.dumps(frame, protocol=5, buffer_callback=self._buffer_callback)
        self.len_buffers.append(self._callback_counter)
        self._callback_counter = 0
        return {"__pickle5_custom__": True, "as_bytes": s_frame}

    def _cpkl_encode(self, obj):
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

    def _pkl_encode(self, obj):
        """
        Encode with pickle library.

        Parameters
        ----------
        obj : object
            Python object.

        Returns
        -------
        dict
            Dictionary with array of serialized bytes.
        """
        return {"__pickle_custom__": True, "as_bytes": pkl.dumps(obj)}

    def _encode_custom(self, obj):
        """
        Serialization hook for msgpack library.

        It encodes complex data types the library couldn`t handle.

        Parameters
        ----------
        obj : object
            Python object.
        """
        if is_pickle5_serializable(obj):
            return self._dataframe_encode(obj)
        elif is_cpkl_serializable(obj):
            return self._cpkl_encode(obj)
        else:
            return self._pkl_encode(obj)

    def serialize(self, data):
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
        process = psutil.Process(os.getpid())
        memory_start = process.memory_full_info()
        result = msgpack.packb(data, default=self._encode_custom)
        memory_end = process.memory_full_info()
        logger.debug(f'SERILALIZITION: {data}')
        logger.debug(f'rss = {memory_end.rss - memory_start.rss} | {memory_start.rss} - {memory_end.rss}')
        logger.debug(f'vms = {memory_end.vms - memory_start.vms} | {memory_start.vms} - {memory_end.vms}')
        logger.debug(f'shared = {memory_end.shared - memory_start.shared} | {memory_start.shared} - {memory_end.shared}')
        logger.debug(f'text = {memory_end.text - memory_start.text} | {memory_start.text} - {memory_end.text}')
        logger.debug(f'data = {memory_end.data - memory_start.data} | {memory_start.data} - {memory_end.data}')
        logger.debug(f'uss = {memory_end.uss - memory_start.uss} | {memory_start.uss} - {memory_end.uss}')
        logger.debug(f'pss = {memory_end.pss - memory_start.pss} | {memory_start.pss} - {memory_end.pss}')
        logger.debug('\n')
        return result

    def _decode_custom(self, obj):
        """
        De-serialization hook for msgpack library.

        It decodes complex data types the library couldn`t handle.

        Parameters
        ----------
        obj : object
            Python object.
        """
        if "__cloud_custom__" in obj:
            return cpkl.loads(obj["as_bytes"])
        elif "__pickle_custom__" in obj:
            return pkl.loads(obj["as_bytes"])
        elif "__pickle5_custom__" in obj:
            frame = pkl.loads(obj["as_bytes"], buffers=self.buffers)
            del self.buffers[: self.len_buffers.pop(0)]
            return frame
        else:
            return obj

    def deserialize(self, s_data):
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
        gc.disable()  # Performance optimization for msgpack
        unpacked_data = msgpack.unpackb(s_data, object_hook=self._decode_custom)
        gc.enable()
        return unpacked_data


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
