import json
from io import BytesIO
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from bokeh.driving import count
from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.plotting import figure
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature

from monitor.apps import Index
from monitor.asset import Static
from monitor.bokeh import bokeh_dynamic
from monitor.dashboard import Dashboard
from monitor.html import CDNs, Component, Element
from monitor.log import Log, time_ms, Notify


class LATEX(Component):
    def __init__(self, text):
        super().__init__()
        self.text = text

    @property
    def heads(self):
        return CDNs.KaTeX + CDNs.KaTeX_auto_render

    @property
    def element(self):
        return self.text


test_board = Dashboard('Classifier Variables')


@test_board.add('Bokeh')
@bokeh_dynamic
def bkapp(doc):
    df = sea_surface_temperature.copy()
    source = ColumnDataSource(data=df)

    plot = figure(x_axis_type='datetime', y_range=(0, 25), y_axis_label='Temperature (Celsius)',
                  title="Sea Surface Temperature at 43.18, -70.43")
    plot.line('time', 'temperature', source=source)

    def callback(attr, old, new):
        if new == 0:
            data = df
        else:
            data = df.rolling(f"{new}D").mean()
        source.data = ColumnDataSource.from_df(data)

    slider = Slider(start=0, end=30, value=0, step=1,
                    title="Smoothing by N Days")
    slider.on_change('value', callback)

    doc.add_root(column(slider, plot))


@test_board.add('Bokeh2')
@bokeh_dynamic
def stock(doc):
    np.random.seed(1)

    MA12, MA26, EMA12, EMA26 = '12-tick Moving Avg', '26-tick Moving Avg', '12-tick EMA', '26-tick EMA'

    source = ColumnDataSource(dict(
        time=[], average=[], low=[], high=[], open=[], close=[],
        ma=[], macd=[], macd9=[], macdh=[], color=[],
    ))

    p = figure(height=500, tools="xpan,xwheel_zoom,xbox_zoom,reset",
               x_axis_type=None, y_axis_location="right")
    p.x_range.follow = "end"
    p.x_range.follow_interval = 100
    p.x_range.range_padding = 0

    p.line(x='time', y='average', alpha=0.2,
           line_width=3, color='navy', source=source)
    p.line(x='time', y='ma', alpha=0.8, line_width=2,
           color='orange', source=source)
    p.segment(x0='time', y0='low', x1='time', y1='high',
              line_width=2, color='black', source=source)
    p.segment(x0='time', y0='open', x1='time', y1='close',
              line_width=8, color='color', source=source)

    p2 = figure(height=250, x_range=p.x_range,
                tools="xpan,xwheel_zoom,xbox_zoom,reset", y_axis_location="right")
    p2.line(x='time', y='macd', color='red', source=source)
    p2.line(x='time', y='macd9', color='blue', source=source)
    p2.segment(x0='time', y0=0, x1='time', y1='macdh', line_width=6,
               color='black', alpha=0.5, source=source)

    mean = Slider(title="mean", value=0, start=-0.01, end=0.01, step=0.001)
    stddev = Slider(title="stddev", value=0.04, start=0.01, end=0.1, step=0.01)
    mavg = Select(value=MA12, options=[MA12, MA26, EMA12, EMA26])

    def _create_prices(t):
        last_average = 100 if t == 0 else source.data['average'][-1]
        returns = np.asarray(np.random.lognormal(mean.value, stddev.value, 1))
        average = last_average * np.cumprod(returns)
        high = average * np.exp(abs(np.random.gamma(1, 0.03, size=1)))
        low = average / np.exp(abs(np.random.gamma(1, 0.03, size=1)))
        delta = high - low
        open = low + delta * np.random.uniform(0.05, 0.95, size=1)
        close = low + delta * np.random.uniform(0.05, 0.95, size=1)
        return open[0], high[0], low[0], close[0], average[0]

    def _moving_avg(prices, days=10):
        if len(prices) < days:
            return [100]
        return np.convolve(prices[-days:], np.ones(days, dtype=float), mode="valid") / days

    def _ema(prices, days=10):
        if len(prices) < days or days < 2:
            return [prices[-1]]
        a = 2.0 / (days+1)
        kernel = np.ones(days, dtype=float)
        kernel[1:] = 1 - a
        kernel = a * np.cumprod(kernel)
        # The 0.8647 normalizes out that we stop the EMA after a finite number of terms
        return np.convolve(prices[-days:], kernel, mode="valid") / (0.8647)

    @count()
    def update(t):
        open, high, low, close, average = _create_prices(t)
        color = "green" if open < close else "red"

        new_data = dict(
            time=[t],
            open=[open],
            high=[high],
            low=[low],
            close=[close],
            average=[average],
            color=[color],
        )

        close = source.data['close'] + [close]
        ma12 = _moving_avg(close[-12:], 12)[0]
        ma26 = _moving_avg(close[-26:], 26)[0]
        ema12 = _ema(close[-12:], 12)[0]
        ema26 = _ema(close[-26:], 26)[0]

        if mavg.value == MA12:
            new_data['ma'] = [ma12]
        elif mavg.value == MA26:
            new_data['ma'] = [ma26]
        elif mavg.value == EMA12:
            new_data['ma'] = [ema12]
        elif mavg.value == EMA26:
            new_data['ma'] = [ema26]

        macd = ema12 - ema26
        new_data['macd'] = [macd]

        macd_series = source.data['macd'] + [macd]
        macd9 = _ema(macd_series[-26:], 9)[0]
        new_data['macd9'] = [macd9]
        new_data['macdh'] = [macd - macd9]

        source.stream(new_data, 300)

    doc.add_root(column(row(mean, stddev, mavg), gridplot(
        [[p], [p2]], toolbar_location="left", width=1000)))
    doc.add_periodic_callback(update, 50)


test_board.add('QED', LATEX(
    R'''\(\mathcal{L_{\mathrm{QED}}}=\bar{\psi}\left(i \cancel{D} -m\right) \psi-\frac{1}{4} F_{\mu \nu} F^{\mu \nu}\)'''))
test_board.add('Lipsum', LATEX(
    R'''
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Pellentesque a leo et metus maximus ullamcorper. Maecenas varius lacinia nisi. Suspendisse vitae purus nisl. Donec ut dui eget dolor ultrices sagittis et eget odio. Sed suscipit pellentesque aliquet. Aenean ut cursus tortor. Nam a molestie odio, rhoncus facilisis sem. Etiam quam turpis, placerat bibendum luctus id, blandit vel libero. Fusce eget ante a urna aliquam aliquam eu at leo. Sed sit amet elementum leo, sed condimentum nunc. Ut elementum sem tellus, vitae interdum nisl ultricies nec.

Cras dignissim fringilla ante. Ut tempor imperdiet ultricies. Donec ut bibendum mauris. Curabitur ex ante, facilisis in dolor at, posuere vehicula lectus. Donec feugiat ligula nisi, quis posuere risus hendrerit quis. Nullam sit amet turpis ornare, mollis sapien eu, ultrices mi. Sed cursus volutpat vehicula. Donec eu nibh quam. Vivamus euismod ipsum ac neque cursus, at commodo est auctor. Nam euismod sollicitudin aliquet. Praesent lacus neque, interdum in massa ut, molestie convallis urna. '''))


x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
buffer = BytesIO()
plt.savefig(buffer, format='svg')
test_board.add('Matplotlib', Component(
    Element.tag('div', buffer.getvalue().decode())))

logger = Log('Classifier')


def printf(x):
    print(json.dumps(x, indent=4))


if __name__ == '__main__':
    Index('test',
          ('https://gitlab.cern.ch/cms-cmu/coffea4bees/-/tree/master/python/classifier',
           Static.shared().url_for('logo/gitlab.svg')),
          Log=logger,
          Classifier=test_board,
          ).start()
    index = 0
    tags = ['plot', 'training', 'loss', 'data', 'test',
            'validation', 'GPU:1', 'GPU:2', 'io', 'dataframe']
    while True:
        print(index)
        notify = Notify.info
        msg = f'log message {index}'
        tex = False
        ntags = np.random.randint(0, len(tags))
        selected = np.random.choice(tags, ntags, replace=False)
        match (index % 25):
            case 0:
                notify = Notify.success
            case 5:
                notify = Notify.warning
            case 15:
                notify = Notify.error
            case 20:
                msg = R'\(\mathcal{L_{\mathrm{QED}}}=\bar{\psi}\left(i \cancel{D} -m\right) \psi-\frac{1}{4} F_{\mu \nu} F^{\mu \nu}\)'
                tex = True
        logger.msg(msg, time_ms(), *selected, notify=notify, tex=tex)
        index += 1
        sleep(1)
