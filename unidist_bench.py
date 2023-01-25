import pandas as pd
import unidist
import time
import os
import asyncio
from collections import defaultdict
from bench_utils import print_times

times_file_name = "bench.csv"

with open(times_file_name, "w") as bench_file:
    pass

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
iterations_count = 3
times = {}

times["init_before"] = time.time()
unidist.init()
times["init_after"] = time.time()

@unidist.remote
def f(df_list):
    df = df_list[0]
    time_2 = time.time()
    df.min()
    time_3 = time.time()
    max_value = df.max()
    time_4 = time.time()
    return max_value, [None, time_2, time_3, time_4], os.getpid()
    
def process_result(times, pids, size, i, result_ref):
    times[f"get_result_{str(size)}_{i}_before"] = time.time()
    result, get_times, pid = unidist.get(result_ref)
    times[f"get_result_{str(size)}_{i}_after"] = time.time()
    
    times[f"corr_{str(size)}_{i}_before"] = get_times[1]
    times[f"corr_{str(size)}_{i}_after"] = get_times[2]
    
    times[f"max_{str(size)}_{i}_before"] = get_times[2]
    times[f"max_{str(size)}_{i}_after"] = get_times[3]
    
    pids.append(pid)

def main(sizes, iterations_count, times):
    for size in sizes:
        args = []
        data = pd.DataFrame({i: list(range(size // 1024)) for i in range(1024)})
        memory_size = sys.getsizeof(data)
        print(f'SIZE: {size} = {memory_size}B')
        for i in range(iterations_count):
            times[f"unidist.put_{str(size)}_{i}_before"] = time.time()
            entity = unidist.put(data)
            times[f"unidist.put_{str(size)}_{i}_after"] = time.time()
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
# Process times from csv                                            #
#####################################################################
with open("bench.csv", "r") as bench_file:
    iterations = defaultdict(int)
    for line in bench_file:
        comand, size, time_before, time_after = line.split(',')
        i = iterations[f"{comand}_{str(size)}"]
        times[f"{comand}_{str(size)}_{i}_before"] = float(time_before)
        times[f"{comand}_{str(size)}_{i}_after"] = float(time_after)
        iterations[f"{comand}_{str(size)}"] += 1

#####################################################################
# Print results                                                     #
#####################################################################

print_times(times, sizes, type_size, iterations_count)