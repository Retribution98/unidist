# Copyright (C) 2021-2022 Modin authors
#
# SPDX-License-Identifier: Apache-2.0

"""MPI communication interfaces."""

from collections import defaultdict
import socket
import time

try:
    import mpi4py
except ImportError:
    raise ImportError(
        "Missing dependency 'mpi4py'. Use pip or conda to install it."
    ) from None

from unidist.core.backends.mpi.core.serialization import (
    deserialize,
    serialize,
    SimpleDataSerializer,
)
import unidist.core.backends.mpi.core.common as common

# TODO: Find a way to move this after all imports
mpi4py.rc(recv_mprobe=False, initialize=False)
from mpi4py import MPI  # noqa: E402


# Sleep time setting inside the busy wait loop
sleep_time = 0.0001


# Logger configuration
logger = common.get_logger("communication", "communication.log")
is_logger_header_printed = False


def log_operation(op_type, status):
    """
    Log a communication between worker processes.

    Parameters
    ----------
    op_type : unidist.core.backends.mpi.core.common.Operation
        Operation type.
    status : mpi4py.MPI.Status
        Represents the status of a reception operation.
    """
    global is_logger_header_printed
    logger_op_name_len = 15
    logger_worker_count = MPIState.get_instance().world_size

    # write header on first worker
    if (
        not is_logger_header_printed
        and MPIState.get_instance().rank == MPIRank.FIRST_WORKER
    ):
        worker_ids_str = "".join([f"{i}{' '*4}" for i in range(logger_worker_count)])
        logger.debug(f'#{" "*logger_op_name_len}{worker_ids_str}')
        is_logger_header_printed = True

    # Write operation to log
    source_rank = status.Get_source()
    dest_rank = MPIState.get_instance().rank
    op_name = common.get_op_name(op_type)
    space_after_op_name = " " * (logger_op_name_len - len(op_name))
    space_before_arrow = ".   " * (min(source_rank, dest_rank))
    space_after_arrow = "   ." * (logger_worker_count - max(source_rank, dest_rank) - 1)
    arrow_line_str = abs(dest_rank - source_rank)
    # Right arrow if dest_rank > source_rank else left
    if dest_rank > source_rank:
        arrow = f'{".---"*arrow_line_str}>'
    else:
        arrow = f'<{arrow_line_str*"---."}'
    logger.debug(
        f"{op_name}:{space_after_op_name}{space_before_arrow}{arrow}{space_after_arrow}"
    )


class MPIState:
    """
    The class holding MPI information.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator.
    rank : int
        Rank of a process.
    world_sise : int
        Number of processes.
    """

    __instance = None

    def __init__(self, comm, rank, world_sise):
        # attributes get actual values when MPI is initialized in `init` function
        self.comm = comm
        self.rank = rank
        self.world_size = world_sise

    @classmethod
    def get_instance(cls, *args):
        """
        Get instance of this class.

        Parameters
        ----------
        *args : tuple
            Positional arguments to create the instance.
            See the constructor's docstring on the arguments.

        Returns
        -------
        MPIState
        """
        if cls.__instance is None and args:
            cls.__instance = MPIState(*args)
        return cls.__instance


class MPIRank:
    """Class that describes ranks assignment."""

    ROOT = 0
    MONITOR = 1
    FIRST_WORKER = 2


def get_topology():
    """
    Get topology of MPI cluster.

    Returns
    -------
    dict
        Dictionary, containing workers ranks assignments by IP-addresses in
        the form: `{"node_ip0": [rank_2, rank_3, ...], "node_ip1": [rank_i, ...], ...}`.
    """
    mpi_state = MPIState.get_instance()
    comm = mpi_state.comm
    rank = mpi_state.rank

    hostname = socket.gethostname()
    host = socket.gethostbyname(hostname)
    cluster_info = comm.allgather((host, rank))
    topology = defaultdict(list)

    for host, rank in cluster_info:
        if rank not in [MPIRank.ROOT, MPIRank.MONITOR]:
            topology[host].append(rank)

    return dict(topology)


# ---------------------------- #
# Main communication utilities #
# ---------------------------- #


def mpi_send_object(comm, data, dest_rank):
    """
    Send Python object to another MPI rank in a blocking way.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    data : object
        Data to send.
    dest_rank : int
        Target MPI process to transfer data.
    """
    comm.send(data, dest=dest_rank)


def mpi_isend_object(comm, data, dest_rank):
    """
    Send Python object to another MPI rank in a non-blocking way.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    data : object
        Data to send.
    dest_rank : int
        Target MPI process to transfer data.

    Returns
    -------
    object
        A handler to MPI_Isend communication result.
    """
    return comm.isend(data, dest=dest_rank)


def mpi_send_buffer(comm, buffer_size, buffer, dest_rank, type=MPI.CHAR):
    """
    Send buffer object to another MPI rank in a blocking way.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    buffer_size : int
        Buffer size in bytes.
    buffer : object
        Buffer object to send.
    dest_rank : int
        Target MPI process to transfer buffer.
    """
    comm.send(buffer_size, dest=dest_rank)
    comm.Send([buffer, type], dest=dest_rank)


def mpi_recv_buffer(comm, source_rank):
    """
    Receive data buffer.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    source_rank : int
        Communication event source rank.

    Returns
    -------
    object
        Array buffer or serialized object.
    """
    buf_size = comm.recv(source=source_rank)
    s_buffer = bytearray(buf_size)
    comm.Recv([s_buffer, MPI.CHAR], source=source_rank)
    return s_buffer


def mpi_busy_wait_recv(comm, source_rank):
    """
    Wait for receive operation result in a custom busy wait loop.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    source_rank : int
        Source MPI process to receive data.
    """
    req_handle = comm.irecv(source=source_rank)
    while True:
        status, data = req_handle.test()
        if status:
            return data
        else:
            time.sleep(sleep_time)


def recv_operation_type(comm):
    """
    Worker receive operation type interface.

    Busy waits to avoid contention. Receives data from any source.

    Parameters
    ----------
    comm : object
        MPI communicator object.

    Returns
    -------
    unidist.core.backends.mpi.core.common.Operation
        Operation type.
    int
        Source rank.
    """
    status = MPI.Status()
    req_handle = comm.irecv(source=MPI.ANY_SOURCE)
    while True:
        is_ready, op_type = req_handle.test(status=status)
        if is_ready:
            log_operation(op_type, status)
            return op_type, status.Get_source()
        else:
            time.sleep(sleep_time)


# --------------------------------- #
# Communication operation functions #
# --------------------------------- #
def recv_complex_data(comm, source_rank):
    """
        Receive the data that may consist of different user provided complex types, lambdas and buffers.

        The data is de-serialized from received buffer.
    m
        Parameters
        ----------
        comm : object
            MPI communicator object.
        source_rank : int
            Source MPI process to receive data from.

        Returns
        -------
        object
            Received data object from another MPI process.
    """
    s_data = comm.recv(source=source_rank)
    # Start unpacking
    return deserialize(s_data)


def send_complex_data(comm, data, dest_rank, is_serialized=False):
    """
    Send the data that consists of different user provided complex types, lambdas and buffers.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.

    Returns
    -------
    object
        A serialized msgpack data.
    list
        A list of pickle buffers.
    list
        A list of buffers amount for each object.
    """
    if not is_serialized:
        data = serialize(data)
    comm.send(data, dest=dest_rank)
    return data


# ---------- #
# Public API #
# ---------- #


def send_complex_operation(comm, operation_type, operation_data, dest_rank):
    """
    Send operation and data that consist of different user provided complex types, lambdas and buffers.

    The data is serialized with ``unidist.core.backends.mpi.core.ComplexDataSerializer``.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : ``unidist.core.backends.mpi.core.common.Operation``
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.
    """
    # Send operation type
    comm.send(operation_type, dest=dest_rank)
    # Send complex dictionary data
    send_complex_data(comm, operation_data, dest_rank)


def send_simple_operation(comm, operation_type, operation_data, dest_rank):
    """
    Send an operation and standard Python data types.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : unidist.core.backends.mpi.core.common.Operation
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.

    Notes
    -----
    Serialization is a simple pickle.dump in this case.
    """
    # Send operation type
    mpi_send_object(comm, operation_type, dest_rank)
    # Send request details
    mpi_send_object(comm, operation_data, dest_rank)


def recv_simple_operation(comm, source_rank):
    """
    Receive an object of a standard Python data type.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    source_rank : int
        Source MPI process to receive data from.

    Returns
    -------
    object
        Received data object from another MPI process.

    Notes
    -----
    De-serialization is a simple pickle.load in this case
    """
    result = comm.recv(source=source_rank)
    return result


def send_operation(
    comm, operation_type, operation_data, dest_rank, is_serialized=False
):
    """
    Send operation and data that consists of different user provided complex types, lambdas and buffers.

    The data is serialized with ``unidist.core.backends.mpi.core.ComplexDataSerializer``.
    Function works with already serialized data.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : ``unidist.core.backends.mpi.core.common.Operation``
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.
    is_serialized : bool
        `operation_data` is already serialized or not.

    Returns
    -------
    dict or None
        Serialization data for caching purpose.

    Notes
    -----
    Function returns ``None`` if `operation_data` is already serialized,
    otherwise ``dict`` containing data serialized in this function.
    """
    # Send operation type
    mpi_send_object(comm, operation_type, dest_rank)
    # Send operation data
    return send_complex_data(
        comm, operation_data, dest_rank, is_serialized=is_serialized
    )


def isend_complex_operation(
    comm, operation_type, operation_data, dest_rank, is_serialized=False
):
    """
    Send operation and data that consists of different user provided complex types, lambdas and buffers.

    Non-blocking asynchronous interface.
    The data is serialized with ``unidist.core.backends.mpi.core.ComplexDataSerializer``.
    Function works with already serialized data.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : ``unidist.core.backends.mpi.core.common.Operation``
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.
    is_serialized : bool
        `operation_data` is already serialized or not.

    Returns
    -------
    dict and dict or dict and None
        Async handlers and serialization data for caching purpose.

    Notes
    -----
    Function always returns a ``dict`` containing async handlers to the sent MPI operations.
    In addition, ``None`` is returned if `operation_data` is already serialized,
    otherwise ``dict`` containing data serialized in this function.
    """
    # Send operation type
    handlers = []
    h1 = mpi_isend_object(comm, operation_type, dest_rank)
    handlers.append((h1, None))

    s_data = serialize(operation_data)

    if not is_serialized:
        s_data = serialize(s_data)

    # Send message pack bytestring
    h2 = mpi_isend_object(comm, s_data, dest_rank)
    handlers.append((h2, None))
    return handlers, s_data


def send_remote_task_operation(comm, operation_type, operation_data, dest_rank):
    """
    Send operation and data that consist of different user provided complex types, lambdas and buffers.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : ``unidist.core.backends.mpi.core.common.Operation``
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.
    """
    # Send operation type
    mpi_send_object(comm, operation_type, dest_rank)
    # Serialize and send the complex data
    send_complex_data(comm, operation_data, dest_rank)


def send_serialized_operation(comm, operation_type, operation_data, dest_rank):
    """
    Send operation and serialized simple data.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    operation_type : unidist.core.backends.mpi.core.common.Operation
        Operation message type.
    operation_data : object
        Data object to send.
    dest_rank : int
        Target MPI process to transfer data.
    """
    # Send operation type
    mpi_send_object(comm, operation_type, dest_rank)
    # Send request details
    mpi_send_buffer(comm, len(operation_data), operation_data, dest_rank)


def recv_serialized_data(comm, source_rank):
    """
    Receive serialized data buffer.

    The data is de-serialized with ``unidist.core.backends.mpi.core.SimpleDataSerializer``.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    source_rank : int
        Source MPI process to receive data from.

    Returns
    -------
    object
        Received de-serialized data object from another MPI process.
    """
    s_buffer = mpi_recv_buffer(comm, source_rank)
    return SimpleDataSerializer().deserialize_pickle(s_buffer)
