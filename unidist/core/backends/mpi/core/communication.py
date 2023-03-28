# Copyright (C) 2021-2023 Modin authors
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
    ComplexDataSerializer,
    SimpleDataSerializer,
)
import unidist.core.backends.mpi.core.common as common

# TODO: Find a way to move this after all imports
mpi4py.rc(recv_mprobe=False, initialize=False)
from mpi4py import MPI  # noqa: E402
from mpi4py.util import pkl5  # noqa: E402


# Sleep time setting inside the busy wait loop
sleep_time = 0.0001

output_shared_size_threshold = 1024**2

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
        worker_ids_str = "".join([f"{i}\t" for i in range(logger_worker_count)])
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


def init_shared_memory(comm, size):
    info = MPI.Info.Create()
    # info.Set("alloc_shared_noncontig", "true")
    win = MPI.Win.Allocate_shared(size, MPI.BYTE.size, comm=comm, info=info)
    return win.Shared_query(MPIRank.MONITOR)

def reserve_shared_memory(comm, data_id, data, is_serialized=False):
    if is_serialized:
        s_data = data["s_data"]
        raw_buffers = data["raw_buffers"]
        buffer_count = data["buffer_count"]

        reservation_data = _send_reserve_operation_impl(
            comm, data_id, s_data, raw_buffers
        )

        return reservation_data, None
    else:
        reservation_data, serialized_data = send_reserve_operation(
            comm, data_id, data
        )
        return reservation_data, serialized_data

def _send_reserve_operation_impl(comm, data_id, s_data, raw_buffers):
    operation_type = common.Operation.RESERVE_SHARING_MEMORY

    operation_data = {
        "id": data_id,
        "size": len(s_data) + sum([len(buf) for buf in raw_buffers])
    }
    # We use a blocking send here because we have to wait for
    # completion of the communication, which is necessary for the pipeline to continue.
    send_simple_operation(
        comm,
        operation_type,
        operation_data,
        MPIRank.MONITOR,
    )
    firstIndex, lastIndex = mpi_busy_wait_recv(comm, MPIRank.MONITOR)
    return {
        "firstIndex": firstIndex, 
        "lastIndex": lastIndex
    }


def send_reserve_operation(comm, data_id, data):
    serializer = ComplexDataSerializer()
    # Main job
    s_data = serializer.serialize(data)
    # Retrive the metadata
    raw_buffers = serializer.buffers
    buffer_count = serializer.buffer_count

    reservation_data = _send_reserve_operation_impl(comm, data_id, s_data, raw_buffers)

    return reservation_data, {
        "s_data": s_data,
        "raw_buffers": raw_buffers,
        "buffer_count": buffer_count,
    }


# ------------------ #
# Data Info packages #
# ------------------ #

class DataInfoType:
    OWNER_DATA = 0
    SHARED_DATA = 1
    LOCAL_DATA = 2

def get_owner_info(data_id, owner_rank):
    info_package = {}
    info_package["package_type"] = DataInfoType.OWNER_DATA
    info_package["id"] = data_id
    info_package["owner"] = owner_rank
    return info_package
    
def get_data_info(s_data_len, raw_buffers_lens, buffer_count):
    info_package = {}
    info_package["package_type"] = DataInfoType.LOCAL_DATA
    info_package["s_data_len"] = s_data_len
    info_package["raw_buffers_lens"] = raw_buffers_lens
    info_package["buffer_count"] = buffer_count
    return info_package

def get_shared_info(data_id, s_data_len, raw_buffers_lens, buffer_count, first_shared_index):
    info_package = {}
    info_package["package_type"] = DataInfoType.SHARED_DATA
    info_package["id"] = data_id
    info_package["s_data_len"] = s_data_len
    info_package["raw_buffers_lens"] = raw_buffers_lens
    info_package["buffer_count"] = buffer_count
    info_package["first_shared_index"] = first_shared_index
    return info_package

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

    Notes
    -----
    This blocking send is used when we have to wait for completion of the communication,
    which is necessary for the pipeline to continue, or when the receiver is waiting for a result.
    Otherwise, use non-blocking ``mpi_isend_object``.
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


def mpi_send_buffer(comm, buffer_size, buffer, dest_rank):
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

    Notes
    -----
    This blocking send is used when we have to wait for completion of the communication,
    which is necessary for the pipeline to continue, or when the receiver is waiting for a result.
    Otherwise, use non-blocking ``mpi_isend_buffer``.
    """
    comm.send(buffer_size, dest=dest_rank)
    comm.Send([buffer, MPI.CHAR], dest=dest_rank)


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


def mpi_isend_buffer(comm, buffer_size, buffer, dest_rank):
    """
    Send buffer object to another MPI rank in a non-blocking way.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    buffer_size : int
        Buffer size in bytes.
    buffer : object
        Buffer object to send.
    dest_rank : int
        Target MPI process to transfer data.

    Returns
    -------
    object
        A handler to MPI_Isend communication result.
    """
    requests = []
    h1 = comm.isend(buffer_size, dest=dest_rank)
    requests.append((h1, None))
    h2 = comm.Isend([buffer, MPI.CHAR], dest=dest_rank)
    requests.append((h2, buffer))
    return requests


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


def _send_complex_data_impl(comm, s_data, raw_buffers, info_package, dest_rank):
    """
    Send already serialized complex data.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    s_data : bytearray
        Serialized data as bytearray.
    raw_buffers : list
        Pickle buffers list, out-of-band data collected with pickle 5 protocol.
    buffer_count : list
        List of the number of buffers for each object
        to be serialized/deserialized using the pickle 5 protocol.
        See details in :py:class:`~unidist.core.backends.mpi.core.serialization.ComplexDataSerializer`.
    dest_rank : int
        Target MPI process to transfer data.
    """
    comm.send(info_package, dest=dest_rank)
    with pkl5._bigmpi as bigmpi:
        comm.Send(bigmpi(s_data), dest=dest_rank)
        for sbuf in raw_buffers:
            comm.Send(bigmpi(sbuf), dest=dest_rank)


def send_complex_data(comm, data, dest_rank, is_serialized=False):
    """
    Send the data that consists of different user provided complex types, lambdas and buffers in a blocking way.

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

    Notes
    -----
    This blocking send is used when we have to wait for completion of the communication,
    which is necessary for the pipeline to continue, or when the receiver is waiting for a result.
    Otherwise, use non-blocking ``isend_complex_data``.
    """
    result = None
    if is_serialized:
        s_data = data["s_data"]
        raw_buffers = data["raw_buffers"]
        buffer_count = data["buffer_count"]
        info_package = data["info_package"]
    else:
        serializer = ComplexDataSerializer()
        # Main job
        s_data = serializer.serialize(data)
        # Retrive the metadata
        raw_buffers = serializer.buffers
        buffer_count = serializer.buffer_count
        info_package = get_data_info(
            len(s_data), 
            [len(sbuf) for sbuf in raw_buffers],
            buffer_count
        )

        result = {
            "s_data": s_data,
            "raw_buffers": raw_buffers,
            "buffer_count": buffer_count,
            "info_package": info_package,
        }
    
    # MPI comminucation
    _send_complex_data_impl(comm, s_data, raw_buffers, info_package, dest_rank)

    # For caching purpose
    return result


def _isend_complex_data_impl(comm, s_data, raw_buffers, info_package, dest_rank):
    """
    Send serialized complex data.

    Non-blocking asynchronous interface.

    Parameters
    ----------
    comm : object
        MPI communicator object.
    s_data : object
        A serialized msgpack data.
    raw_buffers : list
        A list of pickle buffers.
    buffer_count : list
        List of the number of buffers for each object
        to be serialized/deserialized using the pickle 5 protocol.
        See details in :py:class:`~unidist.core.backends.mpi.core.serialization.ComplexDataSerializer`.
    dest_rank : int
        Target MPI process to transfer data.

    Returns
    -------
    list
        A list of pairs, ``MPI_Isend`` handler and associated data to send.
    """
    handlers = []
    h1 = comm.isend(info_package, dest=dest_rank)
    handlers.append((h1, None))

    with pkl5._bigmpi as bigmpi:
        h2 = comm.Isend(bigmpi(s_data), dest=dest_rank)
        handlers.append((h2, s_data))
        for sbuf in raw_buffers:
            h_sbuf = comm.Isend(bigmpi(sbuf), dest=dest_rank)
            handlers.append((h_sbuf, sbuf))

    return handlers


def isend_complex_data(comm, data, dest_rank):
    """
    Send the data that consists of different user provided complex types, lambdas and buffers in a non-blocking way.

    Non-blocking asynchronous interface.

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
    list
        A list of pairs, ``MPI_Isend`` handler and associated data to send.
    object
        A serialized msgpack data.
    list
        A list of pickle buffers.
    list
        A list of buffers amount for each object.
    """
    handlers = []

    serializer = ComplexDataSerializer()
    # Main job
    s_data = serializer.serialize(data)
    # Retrive the metadata
    raw_buffers = serializer.buffers
    buffer_count = serializer.buffer_count

    info_package = get_data_info(
        len(s_data), 
        [len(sbuf) for sbuf in raw_buffers],
        buffer_count
    )

    # Send message pack bytestring
    handlers.extend(
        _isend_complex_data_impl(comm, s_data, raw_buffers, info_package, dest_rank)
    )

    return handlers, s_data, raw_buffers, buffer_count, info_package


def recv_complex_buffers(comm, source_rank, info):
    """
    Receive the data that may consist of different user provided complex types, lambdas and buffers.

    The data is de-serialized from received buffer.

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
    msgpack_buffer = bytearray(info["s_data_len"])
    buffer_count = info["buffer_count"]
    raw_buffers = list(map(bytearray, info["raw_buffers_lens"]))
    with pkl5._bigmpi as bigmpi:
        comm.Recv(bigmpi(msgpack_buffer), source=source_rank)
        for rbuf in raw_buffers:
            comm.Recv(bigmpi(rbuf), source=source_rank)

    # Set the necessary metadata for unpacking
    deserializer = ComplexDataSerializer(raw_buffers, buffer_count)

    # Start unpacking
    return deserializer.deserialize(msgpack_buffer)


# ---------- #
# Public API #
# ---------- #


def send_simple_operation(comm, operation_type, operation_data, dest_rank):
    """
    Send an operation type and standard Python data types in a blocking way.

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
    * This blocking send is used when we have to wait for completion of the communication,
    which is necessary for the pipeline to continue, or when the receiver is waiting for a result.
    Otherwise, use non-blocking ``isend_simple_operation``.
    * Serialization of the data to be sent takes place just using ``pickle.dump`` in this case.
    """
    # Send operation type
    mpi_send_object(comm, operation_type, dest_rank)
    # Send the details of a communication request
    mpi_send_object(comm, operation_data, dest_rank)


def isend_simple_operation(comm, operation_type, operation_data, dest_rank, is_blocking_op=False):
    """
    Send an operation type and standard Python data types in a non-blocking way.

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

    Returns
    -------
    list
        A list of pairs, ``MPI_Isend`` handler and associated data to send.

    Notes
    -----
    Serialization of the data to be sent takes place just using ``pickle.dump`` in this case.
    """
    # Send operation type
    handlers = []
    if not is_blocking_op:
        h1 = mpi_isend_object(comm, operation_type, dest_rank)
        handlers.append((h1, operation_type))
    # Send the details of a communication request
    h2 = mpi_isend_object(comm, operation_data, dest_rank)
    handlers.append((h2, operation_data))
    return handlers


def recv_simple_data(comm, source_rank):
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
    return comm.recv(source=source_rank)


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

    # Send operation data
    if is_serialized:
        # Send already serialized data
        s_data = operation_data["s_data"]
        raw_buffers = operation_data["raw_buffers"]
        buffer_count = operation_data["buffer_count"]
        info_package = operation_data["info_package"]

        h2_list = _isend_complex_data_impl(
            comm, s_data, raw_buffers, info_package, dest_rank
        )
        handlers.extend(h2_list)

        return handlers, None
    else:
        # Serialize and send the data
        h2_list, s_data, raw_buffers, buffer_count, info_package = isend_complex_data(
            comm, operation_data, dest_rank
        )
        handlers.extend(h2_list)
        return handlers, {
            "s_data": s_data,
            "raw_buffers": raw_buffers,
            "buffer_count": buffer_count,
            "info_package": info_package
        }


def isend_serialized_operation(comm, operation_type, operation_data, dest_rank):
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

    Returns
    -------
    list
        A list of pairs, ``MPI_Isend`` handler and associated data to send.
    """
    handlers = []
    # Send operation type
    h1 = mpi_isend_object(comm, operation_type, dest_rank)
    handlers.append((h1, operation_type))
    # Send the details of a communication request
    h2_list = mpi_isend_buffer(comm, len(operation_data), operation_data, dest_rank)
    handlers.extend(h2_list)
    return handlers


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
