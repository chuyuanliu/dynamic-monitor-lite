from functools import partial
from typing import Callable, TypeVar, overload

from flask import Blueprint, Flask
from flask_socketio import SocketIO, emit

from .address import URL, absolute
from .utils import bp_new, bp_register

__all__ = [
    'API',
    'api',
]

_API = 'api'
_AJAX = 'ajax'
_SSE = 'sse'  # TODO
_WS = 'ws'


class API:
    _socket: SocketIO = None
    _blueprint = bp_new(_API, __name__, url_prefix=absolute(_API))
    _apis: set[tuple[str, ...]] = {(_API,)}

    @overload
    @classmethod
    def register(cls, *route: str) -> str:
        ...

    @overload
    @classmethod
    def register(cls, *route: str, import_name: str) -> tuple[Blueprint, str]:
        ...

    @classmethod
    def register(
            cls,
            *route: str,
            import_name: str = None):
        route = URL.route(_API, *route)
        url = absolute(*route)
        if route in cls._apis:
            raise ValueError(
                f'{url} is already registered')
        else:
            cls._apis.add(route)
            if import_name is None:
                return url
            else:
                blueprint = bp_new(
                    '_'.join(route),
                    import_name,
                    url_prefix=absolute(*route[1:]))
                bp_register(cls._blueprint, blueprint)
                return blueprint, url

    @classmethod
    def ajax(cls, *route: str, import_name: str):
        return cls.register(_AJAX, *route, import_name=import_name)

    @classmethod
    def ws(cls, *route: str):
        return WebSocket(cls.register(_WS, *route))

    @classmethod
    def socket(cls, app: Flask):
        cls._socket = SocketIO(app)
        for event, route, handler in WebSocket._event_handlers:
            cls._socket.on_event(event, handler, namespace=route)
        return cls._socket


_SocketEventHandlerT = TypeVar('_SocketEventHandlerT')


class WebSocket:
    _event_handlers: list[tuple[str, str, Callable]] = []

    def __init__(self, route: str):
        self.route = route

    def emit(self, event: str, data, broadcast: bool = False):
        if broadcast:
            API._socket.emit(event, data, namespace=self.route)
        else:
            emit(event, data, namespace=self.route)

    @overload
    def on(self, event: str) -> Callable[[_SocketEventHandlerT], _SocketEventHandlerT]:
        ...

    @overload
    def on(self, event: str, handler: _SocketEventHandlerT) -> _SocketEventHandlerT:
        ...

    def on(self, event: str, handler: _SocketEventHandlerT = None):
        if handler is None:
            return partial(self.on, event)
        self._event_handlers.append((event, self.route, handler))
        return handler


api = API._blueprint
