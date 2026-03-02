"""Functions for running pipelines across multiple tasks."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import os

__all__ = ["run_pipeline_parallel", "run_pipeline_sequential"]


def run_pipeline_sequential(execute, tasks):
    """
    Execute a pipeline across multiple tasks in sequence.

    If any task raises an exception, the exception is re-raised with a note
    indicating the failing task index and arguments.

    Parameters
    ----------
    execute : callable
        A top-level function (must be pickleable) that will be called by
        unpacking the task arguments.
    tasks : list of tuple
        List of argument tuples. Each element of the list is unpacked into
        execute as a task.

    Returns
    -------
    results : list
        List of results in the same order as tasks.

    Raises
    ------
    Exception
        Re-raises any exception raised by a task, adding a note of the failing
        task index and arguments.
    """
    results = [None] * len(tasks)
    for index, task in enumerate(tasks):
        try:
            results[index] = execute(*task)
        except Exception as e:
            message = (
                f"Occurred at task index {index} with arguments{tasks[index]}."
            )
            e.add_note(message)
            raise
    return results


def run_pipeline_parallel(execute, tasks, *, num_workers=None):
    """
    Execute a pipeline across multiple tasks in parallel using a process pool.

    If any task raises an exception:
        - All remaining tasks are cancelled.
        - The process pool is shut down.
        - The exception is re-raised with a note indicating the failing task.

    Parameters
    ----------
    execute : callable
        A top-level function (must be pickleable) that will be called by
        unpacking the task arguments.
    tasks : list of tuple
        List of argument tuples. Each element of the list is unpacked into
        execute as a task.
    num_workers : int or None, optional
        Number of worker processes. If None, defaults to `os.cpu_count()`.

    Returns
    -------
    results : list
        List of results in the same order as tasks.

    Raises
    ------
    Exception
        Re-raises any exception raised by a task, adding a note of the failing
        task index and arguments.
    """
    if num_workers is None:
        num_workers = os.cpu_count()

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(execute, *task): idx
            for idx, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                # Shutting down the executor will not stop any processes that
                # are already running, so it may take some time for the
                # program to fully complete even after an exception is raised.
                executor.shutdown(wait=False, cancel_futures=True)
                message = (
                    f"Occurred at task index {index} with arguments"
                    f"{tasks[index]}."
                )
                e.add_note(message)
                raise
    return results
