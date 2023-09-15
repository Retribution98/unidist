# distutils: language = c++

from libc.stdint cimport uint8_t
cimport memory

import time

def write_to(const uint8_t[:] inband, uint8_t[:] dst, long first_index, int align_size, int memcopy_threads):
    with nogil:
        memory.parallel_memcopy(&dst[first_index],
                                &inband[0],
                                len(inband),
                                align_size,
                                memcopy_threads)
