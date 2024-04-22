from threading import Thread

from flask import Flask

from .address import Address
from .api import API, api
from .asset import asset, Static
from .html import NestedDict, Page
from .utils import bp_register

__all__ = [
    'App',
    'Index'
]

_FLASK = 'flask'


class App(Page):
    options = {
        'head': {
            'title': {
                'format': '{title}',
            },
        }
    }

    def __init__(
            self,
            title: str,
            options: NestedDict,
            *pages: Page):
        super().__init__(title, __name__, None, options)
        self._app = Flask(__name__)
        self._socket = API.socket(self._app)
        self._pages = pages
        self._thread = Thread(target=self._socket.run, kwargs={
            'app': self._app,
            'port': Address.port(_FLASK),
            'debug': False,
            'log_output': False})

        for bp in (asset, api, self.page, *(p.page for p in pages)):
            bp_register(self._app, bp)

    def start(self):
        for page in self._pages:
            page.start()
        self._thread.start()

    def stop(self):
        for page in self._pages:
            page.stop()


class Index(App):
    def __init__(
            self,
            title: str,
            *externals: tuple[str, str],
            options: NestedDict = None,
            **pages: Page):
        super().__init__(title, options, *pages.values())
        self.add_context(
            tabs=pages,
            externals=externals,
            icons=Static.shared().url_for('icon/phosphor-icons.svg'))


Address.port(_FLASK, 8000)  # TODO for test
