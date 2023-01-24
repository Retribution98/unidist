import numpy as np
import unidist
import time

sizes = [
    1024 * 50,
    1024 * 150,
    1024 * 300,
    # 1024**2,
    # 1024**2 * 100,
    # 1024**2 * 500,
    # int(1024**3 * 1.5)
]
iterations_count = 3

unidist.init()
times = {}

@unidist.remote
def f(args):
    for arg in args:
        np.max(arg)
    return 0
    

for size in sizes:
    args = []
    for i in range(iterations_count):
        data = np.zeros(size)
        args.append(unidist.put(data))
    
    unidist.get(f.remote(args))

with open("bench.csv", "r") as bench_file:
    print(bench_file.read())
