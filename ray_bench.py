import numpy as np
import ray
import time

sizes = [
    1024 * 50,
    1024 * 150,
    1024 * 300,
    1024**2,
    1024**2 * 100,
    1024**2 * 500,
    int(1024**3 * 1.5)
]
iterations_count = 3

ray.init()
times = {}

@ray.remote
def f(args):
    times = []
    for arg in args:
        time_1 = time.time()
        data = ray.get(arg)
        time_2 = time.time()
        times.append((time_1, time_2))
        np.max(arg)
    return times
    

for size in sizes:
    args = []
    for i in range(iterations_count):
        data = np.zeros(size)
        times[f"put_{str(size)}_{i}_before"] = time.time()
        entity = ray.put(data)
        times[f"put_{str(size)}_{i}_after"] = time.time()
        args.append(entity)
    
    get_times = ray.get(f.remote(args))
    for i in range(iterations_count):
        t = get_times[i]
        times[f"get_{str(size)}_{i}_before"] = t[0]
        times[f"get_{str(size)}_{i}_after"] = t[1]


#############################################################
# PRINT RESULT                                              #
#############################################################

#header
def get_size_name(size):
    degree = 0
    while size >= 1024:
        degree += 1
        size = size / 1024
    postfix = ""
    if degree == 0:
        postfix = "B"
    elif degree == 1:
        postfix = "KB"
    elif degree == 2:
        postfix = "MB"
    elif degree == 3:
        postfix = "GB"
    elif degree == 4:
        postfix = "TB"
    if size % 1 > 0:
        size = "%.2f" % size
    else:
        size = str(int(size))
    return f'{size} {postfix}'

print(f'\t| {" | ".join([get_size_name(size).ljust(7) for size in sizes])} |')
print(f'{"-"*8}|{"|".join(["-"*9 for _ in sizes])}|')


#body
commands = ["put", "get"]
for command in commands:
    for i in range(iterations_count):
        time_array = []
        for size in sizes:
            name_patern = f'{command}_{str(size)}_{i}'
            time_diff = times[f'{name_patern}_after'] - times[f'{name_patern}_before']
            time_array.append(time_diff)
        first_column = ""
        if i == 0:
            first_column = command.upper()
        print(f'{first_column}\t| {" | ".join("{:.5f}".format(t)[:7] for t in time_array)} |')

    average_times = []
    for size in sizes:
        average_times.append(np.average(
            [times[f"{command}_{str(size)}_{i}_after"] - times[f"{command}_{str(size)}_{i}_before"]
            for i in range(iterations_count)]
        ))
    print(f'\t|{"|".join(["-"*9 for _ in sizes])}|')
    print(f'AVERAGE | {" | ".join("{:.5f}".format(t)[:7] for t in average_times)} |')
    print(f'{"-"*8}|{"|".join(["-"*9 for _ in sizes])}|')