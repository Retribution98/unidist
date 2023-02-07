# Copyright (C) 2021-2022 Modin authors
#
# SPDX-License-Identifier: Apache-2.0

import time
import unidist.core.backends.mpi.core.common as common
import unidist.core.backends.mpi.core.communication as communication

# Logger configuration
# When building documentation we do not have MPI initialized so
# we use the condition to set "worker_0.log" in order to build it succesfully.
log_file = "worker_{}.log".format(
    communication.MPIState.get_instance().rank
    if communication.MPIState.get_instance() is not None
    else 0
)
logger = common.get_logger("worker", log_file)


class AsyncOperations:
    """
    Class that stores MPI async communication handlers.

    Class holds a reference to sending data to prolong data lifetime during send operation.
    """

    __instance = None

    def __init__(self):
        # Cleanup frequency settings
        self._count_threshold = 10**5
        self._time_threshold = 10  # seconds
        self._timestamp = 0  # seconds
        # I-prefixed mpi call handlers
        self._send_async_handlers = []

    @classmethod
    def get_instance(cls):
        """
        Get instance of ``AsyncOperations``.

        Returns
        -------
        AsyncOperations
        """
        if cls.__instance is None:
            cls.__instance = AsyncOperations()
        return cls.__instance

    def extend(self, handlers_list):
        """
        Extend internal list with `handler_list`.

        Parameters
        ----------
        handler_list : list
            A list of pairs with handler and data reference.
        """
        # force cheeck completed requests to avoid stack overflow
        # the value of the criteria is determined experimentally and can be chaned in the future
        timestamp_snapshot = time.perf_counter()
        if (
            timestamp_snapshot - self._timestamp > self._time_threshold
            or len(self._send_async_handlers) > self._count_threshold
        ):
            self.check()
        self._send_async_handlers.extend(handlers_list)

    def check(self):
        """Check all MPI async send requests readiness and remove a reference to sending data."""

        def is_ready(handler):
            is_ready = handler.Test()
            if is_ready:
                logger.debug("CHECK ASYNC HANDLER {} - ready".format(handler))
            else:
                logger.debug("CHECK ASYNC HANDLER {} - not ready".format(handler))
            return is_ready

        # tup[0] - mpi async send handler object
        self._send_async_handlers[:] = [
            tup for tup in self._send_async_handlers if not is_ready(tup[0])
        ]
        # save check time to force check
        self._timestamp = time.time()

    def finish(self):
        """Cancel all MPI async send requests."""
        for handler, data in self._send_async_handlers:
            logger.debug("WAIT ASYNC HANDLER {}".format(handler))
            handler.Cancel()
            handler.Wait()
        self._send_async_handlers.clear()