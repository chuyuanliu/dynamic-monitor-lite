from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint

from .address import URL, absolute
from .utils import bp_new, bp_register, exts

__all__ = [
    'Asset',
    'Static',
    'Resource',
    'asset',
]

_ASSET = 'asset'
_STATIC = 'static'
_RESOURCE = 'resource'  # TODO


class Asset:
    _blueprint = bp_new(
        _ASSET, __name__, url_prefix=absolute(_ASSET), static_folder=_STATIC)
    _assets: dict[tuple[str, ...], str] = {
        (_ASSET,): __name__,
        (_ASSET, _STATIC): __name__
    }
    _exts: dict[str, str] = (
        dict.fromkeys(('svg', 'png', 'jpg'), 'img')
    )

    @classmethod
    def _get_type(cls, file: str, type: str):
        if type is ...:
            type = exts(file)[-1]
            type = cls._exts.get(type, type)
        elif type is None:
            type = ''
        return type


class Static(Asset):
    _shared: Static = None

    def __init__(self, route: tuple[str], blueprint: Blueprint):
        self._route = route
        self._blueprint = blueprint

    def url_for(self, file: str, type: str = ...):
        last_modified = int(self.stat(file, type).st_mtime)
        type = self._get_type(file, type)
        return absolute(*self._route, type, file) + URL.new_query(m=last_modified)

    def open(self, file: str, type: str = ..., mode: str = 'r'):
        type = self._get_type(file, type)
        return open(Path(self._blueprint.static_folder)/type/file, mode=mode)

    def stat(self, file: str, type: str = ...):
        type = self._get_type(file, type)
        return os.stat(Path(self._blueprint.static_folder)/type/file)

    @classmethod
    def register(
            cls,
            import_name: str = __name__,
            *route: str):
        route = URL.route(_ASSET, _STATIC, *route)
        blueprint = bp_new(
            '_'.join(route),
            import_name,
            static_folder=_STATIC,
            static_url_path=absolute(*route[1:]))
        if route not in cls._assets:
            cls._assets[route] = import_name
            bp_register(cls._blueprint, blueprint)
        else:
            registered = cls._assets[route]
            if registered != import_name:
                raise ValueError(
                    f'{absolute(*route)} is already registered to "{registered}"')
        return cls(route, blueprint)

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls.register()
        return cls._shared


class Resource:
    ...  # TODO


asset = Asset._blueprint
