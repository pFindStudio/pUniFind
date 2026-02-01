# Copyright (c) 2024, pUniFind Team
# Progress bar utilities

import sys
from typing import Optional, Iterator, Any
from tqdm import tqdm


class ProgressBar:
    """
    Progress bar wrapper for iterators.
    """

    def __init__(
        self,
        iterable: Iterator,
        log_format: Optional[str] = None,
        log_interval: int = 100,
        prefix: str = "",
        default_log_format: str = "tqdm",
    ):
        self.iterable = iterable
        self.log_format = log_format or default_log_format
        self.log_interval = log_interval
        self.prefix = prefix
        self._pbar = None
        self._current_step = 0

    def __iter__(self):
        if self.log_format == "tqdm":
            self._pbar = tqdm(
                self.iterable,
                desc=self.prefix,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            return iter(self._pbar)
        elif self.log_format == "simple":
            return self._simple_iter()
        elif self.log_format == "none":
            return iter(self.iterable)
        else:
            # Default to tqdm
            self._pbar = tqdm(
                self.iterable,
                desc=self.prefix,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            return iter(self._pbar)

    def _simple_iter(self):
        for i, item in enumerate(self.iterable):
            if i % self.log_interval == 0:
                print(f"{self.prefix} | step {i}", flush=True)
            yield item

    def log(self, stats: dict, step: Optional[int] = None) -> None:
        """
        Log statistics at the current step.

        Args:
            stats: Dictionary of statistics to log
            step: Optional step number
        """
        self._current_step = step if step is not None else self._current_step + 1

        if self._pbar is not None and stats:
            self._pbar.set_postfix(stats)

    def print(self, stats: dict) -> None:
        """
        Print statistics.

        Args:
            stats: Dictionary of statistics to print
        """
        if stats:
            msg = " | ".join(f"{k}: {v}" for k, v in stats.items())
            print(f"{self.prefix} | {msg}", flush=True)


def progress_bar(
    iterable: Iterator,
    log_format: Optional[str] = None,
    log_interval: int = 100,
    prefix: str = "",
    default_log_format: str = "tqdm",
) -> ProgressBar:
    """
    Create a progress bar for an iterable.

    Args:
        iterable: Iterator to wrap
        log_format: Format for logging ('tqdm', 'simple', 'none')
        log_interval: Interval for logging in simple mode
        prefix: Prefix string for the progress bar
        default_log_format: Default format if log_format is None

    Returns:
        ProgressBar instance
    """
    return ProgressBar(
        iterable=iterable,
        log_format=log_format,
        log_interval=log_interval,
        prefix=prefix,
        default_log_format=default_log_format,
    )
