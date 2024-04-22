from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Generic, Optional, TypeVar

from flask import Blueprint, Flask

_RefT = TypeVar('_RefT')
_capitalized_pattern = re.compile(r'[A-Z]+(?![a-z])|[A-Z][a-z]+|[0-9]+')
_lower_underscore_pattern = re.compile(r'[^a-zA-Z0-9]+')
_lower_hyphen_pattern = re.compile(r'[^a-zA-Z0-9/]+')


class ref(Generic[_RefT]):
    def __init__(self, value: _RefT = None):
        self.value = value


def exts(file: str):
    return (*(ext.lstrip('.') for ext in Path(file).suffixes),) or ('',)


def split_capitalized(text: str) -> list[str]:
    return _capitalized_pattern.findall(text)


def lower_underscore(text: str):
    return _lower_underscore_pattern.sub('_', text).lower()


def lower_hyphen(text: str):
    return _lower_hyphen_pattern.sub('-', text).lower()


def bp_new(
        name: str,
        import_name: str,
        url_prefix: Optional[str] = None,
        static_folder: Optional[str | os.PathLike] = None,
        static_url_path: Optional[str] = None,
        template_folder: Optional[str | os.PathLike] = None,
        **kwargs):
    if url_prefix is not None:
        url_prefix = lower_hyphen(url_prefix)
    if static_url_path is not None:
        static_url_path = lower_hyphen(static_url_path)
    return Blueprint(
        lower_underscore(name),
        import_name,
        url_prefix=url_prefix,
        static_folder=static_folder,
        static_url_path=static_url_path,
        template_folder=template_folder,
        **kwargs)


def bp_register(
        parent: Flask | Blueprint,
        child: Blueprint,
        url_prefix: Optional[str] = None,
        **kwargs):
    if url_prefix is not None:
        url_prefix = lower_hyphen(url_prefix)
    parent.register_blueprint(child, url_prefix=url_prefix, **kwargs)
