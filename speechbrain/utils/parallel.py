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


def parallel_map(
    fn: Callable[[Any], None],
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
    known_len = len(source) if hasattr(source, "__len__") else None
    source_it = iter(source)

    futures = deque()
    cv = Condition()
    just_finished_count = 0

    if progress_bar:
        tqdm_final_kwargs = {"total": known_len}
        tqdm_final_kwargs.update(progress_bar_kwargs)
        pbar = tqdm(**tqdm_final_kwargs)
    else:
        pbar = None

    def _bump_processed_count(future):
        """Notifies the main thread of the finished job, bumping the number of
        jobs it should requeue. Updates the progress bar based on the returned
        chunk length.

        Arguments
        ---------
        future: concurrent.futures.Future
            A future holding a processed chunk (of type `list`).
        """
        nonlocal just_finished_count

        # update progress bar with the length of the output as the progress bar
        # is over element count, not chunk count.
        if pbar is not None:
            pbar.update(len(future.result()))

        # wake up dispatcher thread to refill the queue
        with cv:
            just_finished_count += 1
            cv.notify()

    def _map_all(executor: Executor):
        """Performs all the parallel mapping logic.

        Arguments
        ---------
        executor: concurrent.futures.Executor
            The executor to `.invoke` all jobs on. The executor is NOT shut down
            at the end of processing.
            """
        nonlocal just_finished_count

        def _enqueue_job():
            """Pulls a chunk from the source iterable and submits it to the
            pool; must be run from the main thread.

            Returns
            -------
            `True` if any job was submitted (that is, if there was any chunk
            left to process), `False` otherwise.
            """
            # immediately deplete the input stream of chunk_size elems (or less)
            chunk = list(itertools.islice(source_it, chunk_size))

            # empty chunk? then we finished iterating over the input stream
            if len(chunk) == 0:
                return False

            future = executor.submit(_chunk_process_wrapper, fn, chunk)
            future.add_done_callback(_bump_processed_count)
            futures.append(future)

            return True

        # initial queue fill
        for _ in range(queue_size):
            if not _enqueue_job():
                break

        # consume & requeue logic
        while len(futures) != 0:
            with cv:
                # wait for new work to be ready. if `cv.notify` was called by a
                # worker _after_ the `with cv` block, then we should not wait!
                # a new wakeup may never actually happen (or later, when a
                # different worker finishes).
                while just_finished_count == 0:
                    cv.wait()

                # avoid data race on just_finished_count, we're still guarded by
                # the lock here.
                to_queue_count = just_finished_count
                just_finished_count = 0

            # try to enqueue as many jobs as there were just finished.
            # when the input is finished, the queue will not be refilled.
            for _ in range(to_queue_count):
                if not _enqueue_job():
                    break

            # as we must yield in order, try to deplete as much futures from the
            # start of the list as are currently available.
            while len(futures) != 0 and futures[0].done():
                yield from futures.popleft().result()

        if pbar is not None:
            pbar.close()

    if executor is not None:
        yield from _map_all(executor)
    else:
        with ProcessPoolExecutor(max_workers=process_count) as pool:
            yield from _map_all(pool)
