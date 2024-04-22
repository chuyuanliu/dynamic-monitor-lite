import os
import re
from copy import deepcopy
from itertools import chain
from typing import Iterable, Optional

from .utils import lower_hyphen

__all__ = [
    'Address',
    'Schemes',
    'URL',
    'absolute',
    'concat',
    'localhost',
    'relative',
]


def concat(args: Iterable[str]):
    return chain(*(split(arg) for arg in args))


def split(path: str):
    return (*filter(None, path.split('/')),)


def relative(*paths: str):
    return '/'.join(concat(paths))


def absolute(*paths: str):
    return '/' + relative(*paths)


class Schemes:
    http = 'http'
    ws = 'ws'


class URL:
    _url_pattern = re.compile(
        r'^(?:(?P<scheme>[\w]+):/[/]+)?(?:(?P<host>[^/:]+))?(?::(?P<port>[0-9]+))?(?:(?P<path>[/]+[^?]*))?(?:\?(?P<query>.*))?$')
    _query_pattern = re.compile(r'(?P<key>.+)=(?P<value>.+)')

    def __init__(
            self,
            scheme: Optional[str],
            host: Optional[str],
            port: Optional[int],
            path: Optional[str],
            **queries: str | list[str]):
        self.scheme = scheme
        self.host = host or ''
        self.port = port
        self.path = path or ''
        self.queries = queries

    def copy(self):
        return URL(self.scheme, self.host, self.port, self.path, **deepcopy(self.queries))

    @property
    def full_path(self):
        if self.host:
            return relative(self.host, self.path)
        return self.path

    def __str__(self):
        query = self.new_query(**self.queries)
        if self.host:
            scheme = f'{self.scheme}://' if self.scheme is not None else ''
            port = f':{self.port}' if self.port is not None else ''
            return f'{scheme}{relative(f"{self.host}{port}", self.path)}{query}'
        return f'{self.path}{query}'

    @classmethod
    def parse(cls, url: str, is_relative: bool = False):
        match = cls._url_pattern.match(url)
        if match is None:
            raise ValueError(f'Invalid URL: {url}')
        groups = match.groupdict()
        query = cls.parse_query(groups['query'] or '')
        if not is_relative:
            port = groups['port']
            if port is not None:
                port = int(port)
            return cls(groups['scheme'], groups['host'], port, groups['path'], **query)
        else:
            return cls(None, None, None, relative(groups['host'] or '', groups['path'] or ''), **query)

    @classmethod
    def parse_query(cls, query: str) -> dict[str, str | list[str]]:
        query = query.lstrip('?')
        queries = {}
        for arg in query.split('&'):
            match = cls._query_pattern.match(arg)
            if match:
                k, v = match.groups()
                v = v.split('+')
                if len(v) == 1:
                    v = v[0]
                queries[k] = v
        return queries

    @classmethod
    def new_query(cls, **queries):
        if not queries:
            return ''
        query_args = []
        for k in sorted(queries):
            v = queries[k]
            if v is None:
                continue
            elif isinstance(v, list):
                query_args.append(f'{k}={"+".join(map(str, v))}')
            else:
                query_args.append(f'{k}={v}')
        return f'?{"&".join(query_args)}'

    @classmethod
    def route(cls, *route: str):
        return (*concat(lower_hyphen(r) for r in route),)


class Address:
    _ports: dict[str, int] = {}
    _assigned_ports: set[int] = set()

    def __init__(
            self,
            name: str = None,
            url: str | tuple[str, int] | URL = None):
        if name is None and url is None:
            raise ValueError('Either local or remote must be specified')
        self.local = None
        self.remote = None
        self.url = None
        self.url_parsed = None
        if name is not None:
            if os.name == 'posix':
                self.local = name  # TODO test
            elif os.name == 'nt':
                self.local = Rf'\\.\pipe\{name}'  # TODO test
        if url is not None:
            if isinstance(url, URL):
                self.url = url.copy()
            else:
                self.url = URL.parse(url if isinstance(url, str) else url[0])
                if isinstance(url, tuple):
                    self.url.port = url[1]
            self.remote = (self.url.full_path, self.url.port)

    def __str__(self):
        address = self.local
        if address is None:
            address = str(self.url)
        return address

    @classmethod
    def port(cls, name: str, port: int = None):
        if port is None:
            return cls._ports[name]
        if name in cls._ports:
            old = cls._ports[name]
            if old != port:
                raise ValueError(
                    f'"{name}" is already listening on port {old}, given a new port {port}')
        else:
            if port in cls._assigned_ports:
                raise ValueError(f'Port {port} is already assigned')
            cls._ports[name] = port
            cls._assigned_ports.add(port)
        return port

    @classmethod
    def release_port(cls, name: str):
        cls._assigned_ports.remove(cls._ports.pop(name))


def localhost(port: int = None, path: str = None, scheme: str = None):
    return str(Address(
        url=URL(
            scheme=scheme,
            host='localhost',
            port=port,
            path=path)))
