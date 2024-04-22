from __future__ import annotations

import time
from enum import Flag
from threading import Lock
from typing import TypedDict

from ..address import Address, Schemes, localhost
from ..api import API
from ..asset import Static
from ..html import NestedDict, Page

__all__ = [
    'Log',
    'Notify',
    'time_ms',
]

_FLASK = 'flask'
_LOG = 'log'


def time_ms():
    return time.time_ns() // 1_000_000


class Notify(Flag):
    info = 0b1
    success = 0b10
    warning = 0b100
    error = 0b1000


class _Msg(TypedDict):
    idx: int
    time: int
    tex: bool
    tags: list[str]
    msg: str


class Log(Page):
    options = {
        'package': {
            'Selectize': {
                'enable': True,
            },
            'Socket.IO': {
                'enable': True,
            },
            'KaTeX': {
                'enable': True,
            }
        },
        'log': {
            'notify': Notify.success | Notify.error,
        }
    }

    _preserved = {n.name for n in Notify}

    def __init__(self, title: str, options: NestedDict = None):
        super().__init__(title, __name__, options=options)

        self._data: list[_Msg] = []
        self._socket = API.ws(_LOG, title)
        self._lock = Lock()

        @self._socket.on('log_fetch')
        def log_fetch(start: int | list[int], end: int = None):
            if isinstance(start, int):
                msgs = self._data[start:end]
            elif isinstance(start, list):
                msgs = [self._data[i] for i in sorted(start)]
            else:
                msgs = None
            if msgs:
                self._socket.emit('log_update', msgs)

        self.add_heads(self._static, 'label.css')
        self.add_context(
            socket_url=localhost(
                Address.port(_FLASK), self._socket.route, Schemes.ws),
            notification_icons={
                k.name: self._static.url_for(f'{k.name}.svg') for k in Notify},
            notification_states={
                k.name: k in self.options['log']['notify'] for k in Notify},
            icons=Static.shared().url_for('icon/phosphor-icons.svg'))

    def msg(self, msg, time_ms: int, *labels: str, notify: Notify = Notify.info, tex: bool = False):
        with self._lock:
            _msg = {
                'index': len(self._data),
                'timestamp': time_ms,
                'tex': tex,
                'label': [notify.name] + sorted(set(labels) - self._preserved),
                'message': str(msg),
            }
            self._data.append(_msg)
        self._socket.emit('log_append', _msg, broadcast=True)
