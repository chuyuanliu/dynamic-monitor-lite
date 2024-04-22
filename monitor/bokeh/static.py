from typing import Callable, Iterable

from bokeh.embed import components
from bokeh.embed.util import submodel_has_python_callbacks
from bokeh.model import Model

from ..html import Component, Element, Head
from .cdn import CDNs


def bokeh_static(doc: Callable[[], Model | Iterable[Model]]):
    models = doc()
    if isinstance(models, Model):
        models = (models,)
    return StaticDoc(*models)


class StaticDoc(Component):
    def __init__(self, *models: Model):
        super().__init__()
        if submodel_has_python_callbacks(models):
            raise RuntimeError(
                'Python callbacks are not supported when exporting to static bokeh components.')
        self._script, divs = components(models)
        self._elements = Element('\n'.join(divs))

    @property
    def heads(self):
        yield from CDNs
        yield Head(self._script)

    @property
    def element(self):
        return self._elements
