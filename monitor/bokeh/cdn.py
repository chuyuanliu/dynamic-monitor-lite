from bokeh.resources import CDN

from ..html import Head

CDNs = (
    Head.cdn(
        url=url,
        integrity=f'sha384-{integrity}',
    )
    for url, integrity in CDN.hashes.items()
)
