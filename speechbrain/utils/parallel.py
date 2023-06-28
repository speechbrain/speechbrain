"""Parallel processing tools to help speed up certain tasks like data
preprocessing.

Authors
 * Sylvain de Langen 2023
"""

import itertools
import multiprocessing
from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor
from threading import Condition
from typing import Any, Callable, Iterable, Optional

from tqdm.auto import tqdm


def _chunk_process_wrapper(fn, chunk):
    return list(map(fn, chunk))


class CancelFuturesOnExit:
    """Context manager that .cancel()s all elements of a list upon exit.
    This is used to abort futures faster when raising an exception."""

    def __init__(self, future_list):
        self.future_list = future_list

    def __enter__(self):
        pass

    def __exit__(self, _type, _value, _traceback):
        for future in self.future_list:
            future.cancel()


class _ParallelMapper:
    """Internal class for `parallel_map`, arguments match the constructor's."""

    def __init__(
        self,
        fn: Callable[[Any], Any],
        source: Iterable[Any],
        process_count: int,
        chunk_size: int,
        queue_size: int,
        executor: Optional[Executor],
        progress_bar: bool,
        progress_bar_kwargs: dict,
    ):
        self.future_chunks = deque()
        self.cv = Condition()
        self.just_finished_count = 0
        """Number of jobs that were just done processing, guarded by
        `self.cv`."""
        self.remote_exception = None
        """Set by a worker when it encounters an exception, guarded by
        `self.cv`."""

        self.fn = fn
        self.source = source
        self.process_count = process_count
        self.chunk_size = chunk_size
        self.queue_size = queue_size
        self.executor = executor

        self.known_len = len(source) if hasattr(source, "__len__") else None
        self.source_it = iter(source)
        self.depleted_source = False

        if progress_bar:
            tqdm_final_kwargs = {"total": self.known_len}
            tqdm_final_kwargs.update(progress_bar_kwargs)
            self.pbar = tqdm(**tqdm_final_kwargs)
        else:
            self.pbar = None

    def run(self):
        """Spins up an executor (if none were provided), then yields all
        processed chunks in order."""
        with CancelFuturesOnExit(self.future_chunks):
            if self.executor is not None:
                # just use the executor we were provided
                yield from self._map_all()
            else:
                # start and shut down a process pool executor -- ok for
                # long-running tasks
                with ProcessPoolExecutor(
                    max_workers=self.process_count
                ) as pool:
                    self.executor = pool
                    yield from self._map_all()

    def _bump_processed_count(self, future):
        """Notifies the main thread of the finished job, bumping the number of
        jobs it should requeue. Updates the progress bar based on the returned
        chunk length.

        Arguments
        ---------
        future: concurrent.futures.Future
            A future holding a processed chunk (of type `list`).
        """
        if future.cancelled():
            # the scheduler wants us to stop or something else happened, give up
            return

        future_exception = future.exception()

        # wake up dispatcher thread to refill the queue
        with self.cv:
            if future_exception is not None:
                # signal to the main thread that it should raise
                self.remote_exception = future_exception

            self.just_finished_count += 1
            self.cv.notify()

        if future_exception is None:
            # update progress bar with the length of the output as the progress
            # bar is over element count, not chunk count.
            if self.pbar is not None:
                self.pbar.update(len(future.result()))

    def _enqueue_job(self):
        """Pulls a chunk from the source iterable and submits it to the
        pool; must be run from the main thread.

        Returns
        -------
        `True` if any job was submitted (that is, if there was any chunk
        left to process), `False` otherwise.
        """
        # immediately deplete the input stream of chunk_size elems (or less)
        chunk = list(itertools.islice(self.source_it, self.chunk_size))

        # empty chunk? then we finished iterating over the input stream
        if len(chunk) == 0:
            self.depleted_source = True
            return False

        future = self.executor.submit(_chunk_process_wrapper, self.fn, chunk)
        future.add_done_callback(self._bump_processed_count)
        self.future_chunks.append(future)

        return True

    def _map_all(self):
        """Performs all the parallel mapping logic.

        Arguments
        ---------
        executor: concurrent.futures.Executor
            The executor to `.invoke` all jobs on. The executor is NOT shut down
            at the end of processing.
        """

        # initial queue fill
        for _ in range(self.queue_size):
            if not self._enqueue_job():
                break

        # consume & requeue logic
        while (not self.depleted_source) or (len(self.future_chunks) != 0):
            with self.cv:
                # if `cv.notify` was called by a worker _after_ the `with cv`
                # block last iteration, then `just_finished_count` would be
                # incremented, but this `cv.wait` would not wake up -- skip it.
                while self.just_finished_count == 0:
                    # wait to be woken up by a worker thread, which could mean:
                    # - that a chunk was processed: try to yield any
                    # - that a call failed with an exception: raise it
                    # - nothing; it could be a spurious CV wakeup: keep looping
                    self.cv.wait()

                if self.remote_exception is not None:
                    raise self.remote_exception

                # store the amount to requeue, avoiding data races
                to_queue_count = self.just_finished_count
                self.just_finished_count = 0

            # try to enqueue as many jobs as there were just finished.
            # when the input is finished, the queue will not be refilled.
            for _ in range(to_queue_count):
                if not self._enqueue_job():
                    break

            # yield from left to right as long as there is enough ready
            # e.g. | done | done | !done | done | !done | !done
            # would yield from the first two. we might deplete the entire queue
            # at that point, the `depleted_source` loop check is needed as such.
            while len(self.future_chunks) != 0 and self.future_chunks[0].done():
                yield from self.future_chunks.popleft().result()

        if self.pbar is not None:
            self.pbar.close()


def parallel_map(
    fn: Callable[[Any], Any],
    source: Iterable[Any],
    process_count: int = multiprocessing.cpu_count(),
    chunk_size: int = 8,
    queue_size: int = 128,
    executor: Optional[Executor] = None,
    progress_bar: bool = True,
    progress_bar_kwargs: dict = {"smoothing": 0.02},
):
    """Maps iterable items with a function, processing chunks of items in
    parallel with multiple processes and displaying progress with tqdm.

    Processed elements will always be returned in the original, correct order.
    Unlike `ProcessPoolExecutor.map`, elements are produced AND consumed lazily.

    Arguments
    ---------
    fn
        The function that is called for every element in the source list.
        The output is an iterator over the source list after fn(elem) is called.

    source: Iterable
        Iterator whose elements are passed through the mapping function.

    process_count: int
        The number of processes to spawn. Ignored if a custom executor is
        provided.
        For CPU-bound tasks, it is generally not useful to exceed logical core
        count.
        For IO-bound tasks, it may make sense to as to limit the amount of time
        spent in iowait.

    chunk_size: int
        How many elements are fed to the worker processes at once. A value of 8
        is generally fine. Low values may increase overhead and reduce CPU
        occupancy.

    queue_size: int
        Number of chunks to be waited for on the main process at a time.
        Low values increase the chance of the queue being starved, forcing
        workers to idle.
        Very high values may cause high memory usage, especially if the source
        iterable yields large objects.

    executor: Optional[Executor]
        Allows providing an existing executor (preferably a
        ProcessPoolExecutor). If None (the default), a process pool will be
        spawned for this mapping task and will be shut down after.

    progress_bar: bool
        Whether to show a tqdm progress bar.

    progress_bar_kwargs: dict
        A dict of keyword arguments that is forwarded to tqdm when
        `progress_bar == True`. Allows overriding the defaults or e.g.
        specifying `total` when it cannot be inferred from the source iterable.
    """
    mapper = _ParallelMapper(
        fn,
        source,
        process_count,
        chunk_size,
        queue_size,
        executor,
        progress_bar,
        progress_bar_kwargs,
    )
    yield from mapper.run()
