from __future__ import annotations

from collections import defaultdict
from typing import Any, Union

NestedDict = dict[str, Union['NestedDict', Any]]


class Options(defaultdict):
    def __init__(self, init: NestedDict = None):
        super().__init__(Options)
        if init is not None:
            self.merge(init)

    def __getitem__(self, __key: str) -> Options:
        return super().__getitem__(__key)

    def copy(self) -> Options:
        return Options(self)

    def merge(self, other: NestedDict):
        for key, value in other.items():
            if isinstance(value, dict):
                self[key].merge(value)
            else:
                self[key] = value
