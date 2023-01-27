from mpi4py import MPI
from mpi4py.util import pkl5
import pandas as pd
import numpy as np
import time
from unidist.core.backends.mpi.core.communication import send_complex_data, recv_complex_data
from bench_utils import print_times

def bench_func(name, comm, send_func, recv_func):
    rank = comm.Get_rank()

    iterations_count = 3
    times = {}

    sizes = [
        1024 * 64,
        1024 * 128,
        1024 * 512,
        1024**2,
        1024**2 * 128,
        1024**2 * 512,
        int(1024**3 * 1.5)
    ]
    type_size = 8
    sizes = [s // type_size for s in sizes]

    if rank == 0:
        print(name)
        tag = 0
        data = {}
        for size in sizes:
            # data = pd.DataFrame({i: list(range(size // 1024)) for i in range(1024)})
            data = np.array(range(size))
            send_func(comm, data, 1)
            for i in range(iterations_count):
                times[f"send_{size}_{i}_before"] = time.time()
                req = send_func(comm, data, 1)
                times[f"send_{size}_{i}_after"] = time.time()
                if req is not None:
                    req.wait()

        recv_times = comm.recv()
        times.update(recv_times)
        print_times(times, sizes, type_size, iterations_count, command_list=["send", "recv"])
    elif rank == 1:
        tag = 0
        for size in sizes:
            result = recv_func(comm, 0)
            for i in range(iterations_count):
                times[f"recv_{size}_{i}_before"] = time.time()
                result = recv_func(comm, 0)
                times[f"recv_{size}_{i}_after"] = time.time()

        comm.send(times, dest=0)

def unidist_send(comm, data, dest):
    send_complex_data(comm, data, dest)

def mpi_send(comm, data, dest):
    comm.send(data, dest=dest)

def mpi_isend(comm, data, dest):
    req = comm.isend(data, dest=dest)
    return req

def unidist_recv(comm, source):
    return recv_complex_data(comm, source)

def mpi_recv(comm, source):
    return comm.recv(source=source)

comm = MPI.COMM_WORLD
bench_func("Unidist.complex_data", comm, unidist_send, unidist_recv) 
bench_func("mpi.send", comm, mpi_send, mpi_recv)
bench_func("mpi.isend", comm, mpi_isend, mpi_recv)

comm = pkl5.Intracomm(MPI.COMM_WORLD)
bench_func("pkl5.send", comm, mpi_send, mpi_recv)
bench_func("pkl5.isend", comm, mpi_isend, mpi_recv)
