import numpy as np
import pandas as pd
import os
import ray
import time
import asyncio
import sys
from bench_utils import print_times

sizes = [
    1024 * 64,
    1024 * 128,
    1024 * 512,
    1024**2,
    1024**2 * 128,
    1024**2 * 512,
    int(1024**3 * 1.5)
]
type_size=8
sizes = [s // type_size for s in sizes]
iterations_count = 3
times = {}

times["init_before"] = time.time()
ray.init()
times["init_after"] = time.time()

@ray.remote
def f(df_ref):
    time_1 = time.time()
    df = ray.get(df_ref[0])
    time_2 = time.time()
    df.min()
    time_3 = time.time()
    max_value = df.max()
    time_4 = time.time()
    return max_value, [time_1, time_2, time_3, time_4], os.getpid()


def process_result(times, pids, size, i, result_ref):
    times[f"get_result_{str(size)}_{i}_before"] = time.time()
    result, get_times, pid = ray.get(result_ref)
    times[f"get_result_{str(size)}_{i}_after"] = time.time()
    
    times[f"get_{str(size)}_{i}_before"] = get_times[0]
    times[f"get_{str(size)}_{i}_after"] = get_times[1]
    
    times[f"corr_{str(size)}_{i}_before"] = get_times[1]
    times[f"corr_{str(size)}_{i}_after"] = get_times[2]
    
    times[f"max_{str(size)}_{i}_before"] = get_times[2]
    times[f"max_{str(size)}_{i}_after"] = get_times[3]
    
    pids.append(pid)
    
def main(sizes, iterations_count, times):
    for size in sizes:
        args = []
        # data = pd.DataFrame({i: list(range(1024)) for i in range(size // 1024)})
        data = np.array(size)
        memory_size = sys.getsizeof(data)
        print(f'SIZE: {size} = {memory_size}B')
        for i in range(iterations_count):
            times[f"put_{str(size)}_{i}_before"] = time.time()
            entity = ray.put(data)
            times[f"put_{str(size)}_{i}_after"] = time.time()
            args.append(entity)

        pids = []
        result_refs = []
        for i in range(iterations_count):
            times[f"remote_{str(size)}_{i}_before"] = time.time()
            result_ref = f.remote([args[i]])
            times[f"remote_{str(size)}_{i}_after"] = time.time()
            result_refs.append(result_ref)

        for i in range(iterations_count):
            process_result(times, pids, size, i, result_refs[i])

        print(f'PIDS: {pids}')
        if len(set(pids)) != len(pids):
            print('Not different PIDs!')


main(sizes, iterations_count, times)


#####################################################################
# Print results                                                     #
#####################################################################

print_times(times, sizes, type_size, iterations_count, ["init", "put", "get", "remote", "get_result", "corr", "max"])