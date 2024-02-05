import csv
import logging
import os
import random
import re
from collections import Counter, defaultdict
from typing import List

log = logging.getLogger("turbine-test")


class Stats(Counter):
    """
    Collect and group error messages for a debug report at the end
    """

    def __str__(self):
        stats_keys = [
            "PASSED",
            "FAILED",
            "SKIPPED",
            "XFAILED",
            "NO_FWD",
            "TIMEOUT",
            "CRASHED",
            "TOTAL",
        ]

        return str([(k, self[k]) for k in stats_keys if k in self])


class ErrorAggregatorDict(object):
    """
    Collect and group error messages for a debug report at the end
    """

    def __init__(self):
        super(ErrorAggregatorDict, self).__init__()
        self.errors = defaultdict(list)

    def __len__(self):
        return len(self.errors)

    def __getitem__(self, item: str):
        return self.errors[item]

    def __iadd__(self, other):
        self.update(other)
        return self

    @classmethod
    def single(cls, error, name):
        obj = ErrorAggregatorDict()
        obj.insert(error, name)
        return obj

    def items(self):
        return self.errors

    # insert into dict with error string and test name
    def insert(self, error: str, name: str):
        self.errors[error].append(name)

    def update(self, other):
        if not len(other):
            return

        other_dict = other.items().items()
        for error, names in other_dict:
            self.errors[error] += names

    def print_report(self):
        if not len(self.errors):
            log.info("No exceptions")
            return

        print("\n" + "".join("*" * 80) + "\n" + "EXCEPTIONS" + "\n" + "".join("*" * 80))
        for error, test_names in sorted(self.errors.items(), key=lambda e: len(e[1])):
            extra = ""
            n_tests = len(test_names)
            if n_tests > 15:
                extra = ", ..."
                test_names = test_names[:15]

            print(
                "\033[1;36m"
                + str(n_tests)
                + ": "
                + ", ".join(test_names)
                + extra
                + "\033[0;0m"
            )
            print("\033[1;31m" + error.strip() + "\033[0;0m" + "\n")
            print("-" * 20)
