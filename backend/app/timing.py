"""Small helper for measuring and logging how long a step takes.

Any slow step in the app (loading a file, embedding text, calling the
LLM, etc.) can be wrapped in this context manager to automatically log
how many seconds it took. This makes it easy to see where time is
being spent without adding timing code everywhere by hand.
"""

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def log_time(step_name: str) -> Iterator[None]:
    """Log how long the wrapped block of code took to run.

    Args:
        step_name: A short, human-readable description of the step,
            used in the log message (e.g. "embedding the question").

    Example:
        with log_time("embedding the question"):
            vector = embeddings.embed_query(text)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s took %.2f s", step_name, elapsed)
