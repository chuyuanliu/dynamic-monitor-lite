from __future__ import annotations

from inspect import getmro
from itertools import chain
from typing import Iterable, Protocol, overload, runtime_checkable

from flask import render_template

from ..asset import Static
from ..utils import bp_new, lower_underscore, split_capitalized
from .cdn import CDNs
from .html import Element, Head, Metas
from .utils import NestedDict, Options

_TEMPLATES = 'templates'
_INDEX = 'index.html'


@runtime_checkable
class URLGenerator(Protocol):
    def url_for(self, file: str, type: str = ...) -> str:
        ...


class _Module:
    def __init_subclass__(cls):
        cls.module_title = ' '.join(split_capitalized(cls.__name__))
        cls.module_name = lower_underscore(cls.module_title)

    def __init__(self):
        self._heads: dict[Head, None] = {}

    def start(self):
        ...

    def stop(self):
        ...

    @overload
    def add_heads(self, *heads: Head):
        ...

    @overload
    def add_heads(self, source: URLGenerator, *files: str):
        ...

    def add_heads(self, *args):
        if args:
            if all(isinstance(arg, Head) for arg in args):
                self._heads.update(dict.fromkeys(args))
            elif isinstance(args[0], URLGenerator) and all(isinstance(arg, str) for arg in args[1:]):
                source: URLGenerator = args[0]
                files = args[1:]
                self._heads.update(dict.fromkeys(
                    Head.asset(source.url_for(f))
                    for f in files))
            else:
                raise TypeError(f'Invalid argument: {args}')


class Component(_Module):
    def __init__(self, element: Element = None, *heads: Head):
        super().__init__()
        if element is not None:
            self._element = element
        self.add_heads(*heads)

    @property
    def heads(self) -> Iterable[Head]:
        yield from self._heads.keys()

    @property
    def element(self) -> Element:
        return self._element


class Page(_Module):
    def __init_subclass__(cls):
        super().__init_subclass__()
        if 'options' not in vars(cls):
            cls.options = Options()
        elif not isinstance(cls.options, Options):
            cls.options = Options(cls.options)

    options = Options({
        'head': {
            'meta': {
                'default': True,
            },
            'title': {
                'format': '{module} - {title}'
            },
            'static': {
                'default': True,
                'global': True,
            },
        },
        'package': {
            'jQuery': {
                'enable': True,
            },
            'Selectize': {
                'enable': False,
            },
            'Socket.IO': {
                'enable': False,
            },
            'KaTeX': {
                'enable': False,
                'auto-render': True,
            },
        },
    })

    @classmethod
    def cascade(cls, options: NestedDict):
        opt_seq: list[Options] = []
        if options is not None:
            opt_seq.append(options)
        for base in getmro(cls):
            opt = vars(base).get('options')
            if opt is not None:
                opt_seq.append(opt)
        if len(opt_seq) == 1:
            return opt_seq[0]
        else:
            opt = opt_seq[-1].copy()
            for o in reversed(opt_seq[:-1]):
                opt.merge(o)
            return opt

    def __init__(
            self,
            title: str,
            import_name: str,
            url_prefix: str = ...,
            options: NestedDict = None):
        super().__init__()

        name = f'{self.module_name}_{title}'
        if url_prefix is ...:
            url_prefix = f'/{self.module_name}/{title}'

        self._page = bp_new(
            name,
            import_name,
            url_prefix=url_prefix,
            template_folder=_TEMPLATES)
        self._context = {}
        self._components: list[Component] = []

        @self._page.route('/')
        def index():
            return render_template(
                f'{self.module_name}/{_INDEX}',
                head='\n'.join(map(str, self._heads)),
                **self._context)

        self.options = self.cascade(options)
        self.option = self.options.merge

        if self.options['head']['meta']['default']:
            self.add_heads(*Metas.default)

        if self.options['package']['jQuery']['enable']:
            self.add_heads(*CDNs.jQuery)
        if self.options['package']['Selectize']['enable']:
            self.add_heads(*CDNs.Selectize)
        if self.options['package']['Socket.IO']['enable']:
            self.add_heads(*CDNs.SocketIO)
        if self.options['package']['KaTeX']['enable']:
            self.add_heads(*CDNs.KaTeX)
            if self.options['package']['KaTeX']['auto-render']:
                self.add_heads(*CDNs.KaTeX_auto_render)

        self._static = Static.register(import_name, self.module_name)
        if self.options['head']['static']['global']:
            self.add_heads(
                Static.shared(), 'style.css', 'grid.js', 'utils.js')
        if self.options['head']['static']['default']:
            self.add_heads(
                self._static, f'{self.module_name}.css', f'{self.module_name}.js')

        self.add_context(
            title=self.options['head']['title']['format'].format(
                module=self.module_title,
                title=title))

    @property
    def page(self):
        return self._page

    @property
    def url(self):
        return self._page.url_prefix

    def add_context(self, **kwargs):
        self._context.update(kwargs)

    def add_components(self, *components: Component):
        self._components.extend(components)
        self.add_heads(*chain(*(c.heads for c in components)))

    def start(self):
        for component in self._components:
            component.start()

    def stop(self):
        for component in self._components:
            component.stop()

    @classmethod
    def option(cls, options: NestedDict):
        cls.options.merge(options)
