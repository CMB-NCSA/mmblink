"""Logging utilities."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import multiprocessing
import sys

DEFAULT_LOG_LEVEL = logging.WARNING
DEFAULT_LOG_FORMAT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def configure_logging(
    *,
    level=None,
    format=None,
    date_format=None,
    filename=None,
    logger_name=None,
):
    """Configure log formatting and handlers.

    By default, configuration applies to the root logger unless a name is specified.

    Parameters
    ----------
    level : int or str, optional
        Level to set the logger. If not specified, DEFAULT_LOG_LEVEL is used.
    format : str, optional
        Format string for the logged output. If not specified,
        DEFAULT_LOG_FORMAT is used.
    format : str, optional
        Format string for the date. If not specified, DEFAULT_DATE_FORMAT is
        used.
    filename : str, optional
        Filename to output logs. If not specified, no file handler is added.
    name : str, optional
        Name of the logger to modify. If not specified, the root logger is
        modified.

    Returns
    -------
    logger : logging.Logger
        The modified logger.
    """
    logger = logging.getLogger(logger_name)

    level = DEFAULT_LOG_LEVEL if level is None else level
    format = DEFAULT_LOG_FORMAT if format is None else format
    date_format = DEFAULT_DATE_FORMAT if date_format is None else date_format

    logger.setLevel(level)

    for handler in logger.handlers.copy():
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(fmt=format, datefmt=date_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if filename is not None:
        file_handler = RotatingFileHandler(
            filename, maxBytes=2_000_000, backupCount=10
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

@contextmanager
def parallel_process_log():
    """Context for process-safe logging.

    Sets up a Queue and QueueListener to handle logs from multiple processes.
    The yielded ProcessPoolExecutor factory function will create executors that
    initialize workers to send logs to a configured queue.

    Yields
    ------
    factory : callable returning concurrent.futures.ProcessPoolExecutor
        Factory function for creating ProcessPoolExecutor instances with the
        same parameters as ProcessPoolExecutor.
    """
    root_logger = logging.getLogger()
    root_handlers = root_logger.handlers.copy()
    root_level = root_logger.level

    ctx = multiprocessing.get_context("spawn")
    log_queue = ctx.Queue()

    # Start the listener with root handlers.
    listener = QueueListener(
        log_queue, *root_handlers, respect_handler_level=True
    )
    listener.start()

    # Route parent-process root logs through the queue while the original
    # handlers are owned by the listener thread.
    queue_handler = QueueHandler(log_queue)
    for handler in root_handlers:
        root_logger.removeHandler(handler)
    root_logger.addHandler(queue_handler)

    def _ProcessPoolExecutor(
        max_workers=None, initializer=None, initargs=(), maxtasksperchild=None
    ):
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_worker_initializer,
            initargs=(log_queue, root_level, initializer, initargs),
            max_tasks_per_child=maxtasksperchild
        )

    try:
        yield _ProcessPoolExecutor
    finally:
        # Restore original root handlers.
        root_logger.removeHandler(queue_handler)
        for handler in root_handlers:
            root_logger.addHandler(handler)
        listener.stop()
        log_queue.close()
        log_queue.join_thread()

# Initializer for child processes.
# Must be defined at the module level to be picklable.
def _worker_initializer(queue, level, initializer, initargs):
    root = logging.getLogger()
    for handler in root.handlers.copy():
        root.removeHandler(handler)
        handler.close()
    root.addHandler(QueueHandler(queue))
    root.setLevel(level)
    if initializer:
        initializer(*initargs)
