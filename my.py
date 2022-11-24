import os
import pandas as pd
import time
import pickle
from memory_profiler import profile
from mpi4py import MPI

from unidist.core.backends.mpi.core.communication import (
    send_complex_data,
    recv_complex_data,
)
from unidist.core.backends.mpi.core.serialization import ComplexDataSerializer


@profile
def check_serialize(df):
    serializer = ComplexDataSerializer()

    time_1 = time.time()
    s_data = serializer.serialize(df)
    time_2 = time.time()
    _ = serializer.deserialize(s_data)
    time_3 = time.time()

    buffers = []
    time_4 = time.time()
    data_5 = pickle.dumps(df, protocol=5, buffer_callback=buffers.append)
    time_5 = time.time()
    _ = pickle.loads(data_5, buffers=buffers)
    time_6 = time.time()

    time_7 = time.time()
    data_4 = pickle.dumps(df, protocol=4)
    time_8 = time.time()
    _ = pickle.loads(data_4)
    time_9 = time.time()

    print(f"Serialization time:\t{time_2-time_1}]\t{time_5-time_4}\t{time_8-time_7}")
    print(f"Deserialization time:\t{time_3-time_2}]\t{time_6-time_5}\t{time_9-time_8}")


# @profile
def check_send_communication(comm, df, dest_rank):
    comm.send("Start communication", dest=dest_rank, tag=1)

    time_1 = time.time()
    send_complex_data(comm, df, dest_rank)
    time_2 = time.time()

    time_3 = time.time()
    comm.send(df, dest=dest_rank)
    time_4 = time.time()

    time_5 = time.time()
    send_complex_data(comm, df, dest_rank)
    time_6 = time.time()

    print(f"Send time:\t{time_2-time_1}\t{time_4-time_3}\t{time_6-time_5}")


# @profile
def check_recv_communication(comm, source_rank):
    comm.recv(source=source_rank, tag=1)

    time_1 = time.time()
    _ = recv_complex_data(comm, source_rank)
    time_2 = time.time()

    time_3 = time.time()
    _ = comm.recv(source=source_rank)
    time_4 = time.time()

    time_5 = time.time()
    _ = recv_complex_data(comm, source_rank)
    time_6 = time.time()

    print(f"Recv time:\t{time_2-time_1}\t{time_4-time_3}\t{time_6-time_5}")


if __name__ == "__main__":
    dataset_dir = "/localdisk/benchmark_datasets"
    dataset_path = "taxi/trips_xaa.csv"

    # df = pd.read_csv(os.path.join(dataset_dir, dataset_path), nrows=1000000)
    # check_serialize(df)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        df = pd.read_csv(os.path.join(dataset_dir, dataset_path), nrows=1000000)
        check_send_communication(comm, df, dest_rank=1)
    elif rank == 1:
        check_recv_communication(comm, source_rank=0)
