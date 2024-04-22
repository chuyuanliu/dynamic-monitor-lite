from .html import Head


class CDNs:
    jQuery = (
        Head.cdn(
            url='https://code.jquery.com/jquery-3.7.1.min.js',
            integrity='sha256-/JqT3SQfawRcv/BIHPThkBvs0OEvtFFmqPF/lYI/Cxo='),
    )
    Selectize = (
        Head.cdn(
            url='https://cdnjs.cloudflare.com/ajax/libs/selectize.js/0.15.2/css/selectize.default.min.css',
            integrity='sha512-pTaEn+6gF1IeWv3W1+7X7eM60TFu/agjgoHmYhAfLEU8Phuf6JKiiE8YmsNC0aCgQv4192s4Vai8YZ6VNM6vyQ=='),
        Head.cdn(
            url='https://cdnjs.cloudflare.com/ajax/libs/selectize.js/0.15.2/js/selectize.min.js',
            integrity='sha512-IOebNkvA/HZjMM7MxL0NYeLYEalloZ8ckak+NDtOViP7oiYzG5vn6WVXyrJDiJPhl4yRdmNAG49iuLmhkUdVsQ=='),
    )
    SocketIO = (
        Head.cdn(
            url='https://cdn.socket.io/4.7.3/socket.io.min.js',
            integrity='sha512-uxgT9qpJ2dYbAdKgbmD+b9e5rEta2gULQno8q0mm8t7CNLUwIOm7V3ALywTg6dAIbw9JxGi6aa3RCRLRdSW15Q=='),
    )
    KaTeX = (
        Head.cdn(
            url='https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css',
            integrity='sha512-fHwaWebuwA7NSF5Qg/af4UeDx9XqUpYpOGgubo3yWu+b2IQR4UeQwbb42Ti7gVAjNtVoI/I9TEoYeu9omwcC6g=='),
        Head.cdn(
            url='https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js',
            integrity='sha512-LQNxIMR5rXv7o+b1l8+N1EZMfhG7iFZ9HhnbJkTp4zjNr5Wvst75AqUeFDxeRUa7l5vEDyUiAip//r+EFLLCyA==',
            defer=...),
    )
    KaTeX_auto_render = (
        Head.cdn(
            url='https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js',
            integrity='sha512-iWiuBS5nt6r60fCz26Nd0Zqe0nbk1ZTIQbl3Kv7kYsX+yKMUFHzjaH2+AnM6vp2Xs+gNmaBAVWJjSmuPw76Efg==',
            defer=...,
            onload='renderMathInElement(document.body);'),
    )
