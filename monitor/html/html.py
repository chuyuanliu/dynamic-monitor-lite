from __future__ import annotations

from textwrap import indent

from ..address import URL
from ..utils import exts


class Element:
    void = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
            'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, html: str = None):
        self.html = html

    def __hash__(self):
        return hash(self.html)

    def __eq__(self, other: Element):
        if isinstance(other, Element):
            return self.html == other.html
        return NotImplemented

    def __str__(self):
        return self.html

    def __repr__(self):
        return self.html

    @classmethod
    def tag(cls, _tag: str, *_contents: str, **attributes: str):
        start = f'<{_tag}{cls.parse_attributes(**attributes)}>'
        if _tag in cls.void:
            return cls(start)
        else:
            _contents = '\n'.join(str(c) for c in _contents if c)
            if _contents:
                _contents = f'\n{indent(_contents, "    ")}\n'
            return cls(f'{start}{_contents}</{_tag}>')

    @classmethod
    def parse_attributes(cls, **attributes: str) -> str:
        parsed = []
        for k in sorted(attributes):
            v = attributes[k]
            if v is None:
                continue
            elif v is ...:
                parsed.append(f' {k}')
            elif isinstance(v, dict):
                parsed.append(cls.parse_attributes(
                    {f'{k}-{v_k}': v_v for v_k, v_v in v.items()}))
            else:
                parsed.append(f' {k}="{v}"')
        return ''.join(parsed)


class Head(Element):
    @classmethod
    def script(cls, src: str, **kwargs: str):
        return cls.tag('script', src=src, **kwargs)

    @classmethod
    def style(cls, href: str, **kwargs: str):
        return cls.tag('link', rel='stylesheet', href=href, **kwargs)

    @classmethod
    def meta(cls, **kwargs: str):
        return cls.tag('meta', **kwargs)

    @classmethod
    def asset(cls, url: str, **kwargs: str):
        ext = exts(URL.parse(url).full_path)[-1]
        if ext == 'js':
            return cls.script(src=url, **kwargs)
        elif ext == 'css':
            return cls.style(href=url, **kwargs)
        else:
            raise ValueError(f'Unknown file type: {url}')

    @classmethod
    def cdn(
            cls,
            url: str,
            integrity: str = None,
            crossorigin: str = 'anonymous',
            referrerpolicy: str = 'no-referrer',
            **kwargs: str):
        return cls.asset(
            url,
            integrity=integrity,
            crossorigin=crossorigin,
            referrerpolicy=referrerpolicy,
            **kwargs)


class Metas:
    utf_8 = Head.meta(
        charset='utf-8')
    viewport = Head.meta(
        name='viewport', content='width=device-width, initial-scale=1')

    default = (utf_8, viewport)
