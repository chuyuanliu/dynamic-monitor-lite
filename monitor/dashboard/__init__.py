from __future__ import annotations

from collections import defaultdict
from typing import Callable, TypeVar, overload

from ..html import Component, Element, NestedDict, Page
from ..utils import lower_hyphen

__all__ = [
    'Dashboard'
]

_ComponentT = TypeVar('_ComponentT')


class Dashboard(Page):
    def __init__(self, title: str, options: NestedDict = None):
        super().__init__(title, __name__, options=options)
        self._titles: dict[str, str] = {}
        self._blocks: dict[tuple[str, str], list[Element]] = defaultdict(list)
        self.add_context(nav=self._blocks)

    @overload
    def add(self, title: str) -> Callable[[_ComponentT], _ComponentT]:
        ...

    @overload
    def add(self, title: str, *components: Component) -> None:
        ...

    def add(self, title: str, *components: Component):
        if not components:
            def wrapper(component):
                self.add(title, component)
                return component
            return wrapper
        _id = lower_hyphen(title)
        if _id in self._titles:
            _title = self._titles[_id]
            if _title != title:
                raise ValueError(
                    f'"{title}" and "{_title}" have the same id="{_id}"')
        else:
            self._titles[_id] = title
        self._blocks[(_id, title)].extend(c.element for c in components)
        self.add_components(*components)
