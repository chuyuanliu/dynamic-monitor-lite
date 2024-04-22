from threading import Thread
from typing import Callable, TypeVar

from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from ..address import Address, Schemes, localhost
from ..api import API
from ..html import Component, Element
from ..utils import ref
from .cdn import CDNs

_BOKEH = 'bokeh'
_FLASK = 'flask'


_LayoutT = TypeVar('_LayoutT')


def bokeh_dynamic(doc: _LayoutT) -> _LayoutT:
    return DynamicDoc(doc)


class DynamicDoc(Component):
    _server = ref[tuple[Server, Thread]]()
    _apps: dict[str, Callable[[Document], None]] = {}

    def __init__(self, doc: Callable[[Document], None]):
        super().__init__()
        self._api = API.register(_BOKEH, doc.__name__)
        self._doc = doc
        self._apps[self._api] = doc

    def __getattr__(self, name: str):
        return getattr(self._doc, name)

    @property
    def heads(self):
        yield from CDNs

    @property
    def element(self):
        return Element(server_document(localhost(Address.port(_BOKEH), self._api, Schemes.http), resources=None))

    @classmethod
    def start(cls):
        if cls._server.value is None:
            server = Server(
                cls._apps,
                io_loop=IOLoop(),
                port=Address.port(_BOKEH),
                allow_websocket_origin=[localhost(Address.port(_FLASK))])
            thread = Thread(target=server.io_loop.start)
            server.start()
            thread.start()
            cls._server.value = (server, thread)

    @classmethod
    def stop(cls):
        if cls._server.value is not None:
            server, thread = cls._server.value
            server.io_loop.stop()
            server.stop()
            thread.join()
            cls._server.value = None


Address.port(_BOKEH, 5006)  # TODO for test
