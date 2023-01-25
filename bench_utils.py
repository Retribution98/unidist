import numpy as np

def print_times(times, sizes, type_size, iterations_count):
    """
    Print time results
    """
    sizes = [size * type_size for size in sizes]

    #header
    first_column_width = 10
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

    print(f'{" "*first_column_width}| {" | ".join([get_size_name(size).ljust(7) for size in sizes])} |')
    print(f'{"-"*first_column_width}|{"|".join(["-"*9 for _ in sizes])}|')


    #body
    commands = ["init", "put", "get", "remote", "get_result", "corr", "max"]
    for command in commands:
        for i in range(iterations_count):
            time_array = []
            for size in sizes:
                name_patern = f'{command}_{str(size)}_{i}' if command != "init" else command
                time_diff = times[f'{name_patern}_after'] - times[f'{name_patern}_before']
                time_array.append(time_diff)
            first_column = ""
            if i == 0:
                first_column = command.upper()
            print(f'{first_column.ljust(first_column_width)}| {" | ".join("{:.5f}".format(t)[:7] for t in time_array)} |')
            
            if command == "init":
                break
        average_times = []
        for size in sizes:
            size_times = []
            for i in range(iterations_count):
                name_patern = f'{command}_{str(size)}_{i}' if command != "init" else command
                size_times.append(times[f"{name_patern}_after"] - times[f"{name_patern}_before"])
            average_times.append(np.average(size_times))
        print(f'{" "*first_column_width}|{"|".join(["-"*9 for _ in sizes])}|')
        print(f'{"AVERAGE".ljust(first_column_width)}| {" | ".join("{:.5f}".format(t)[:7] for t in average_times)} |')
        print(f'{"-"*first_column_width}|{"|".join(["-"*9 for _ in sizes])}|')