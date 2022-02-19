import datetime
import json
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatch
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import requests

from svg2path import parse_file


class Forecast:
    
    def __init__(self, code: str = None, city: str = None,
                 update_time: datetime.datetime = None):
        self.code = code
        self.city = city
        self.update_time = update_time
        self.data = [] # type: List[ForecastDay]
        
    @classmethod
    def from_nmc(cls, code: str, city: str = None) -> 'Forecast':
        url = f'http://www.nmc.cn/rest/weather?stationid={code}'
        req = requests.get(url, timeout=10)
        res = req.json()
        with open('cache.json', 'w', encoding='utf8') as fout:
            json.dump(res, fout, ensure_ascii=False)
        return cls.from_nmc_json(res, city=city)

    @classmethod
    def from_nmc_json(cls, nmc_json: dict, city: str = None) -> 'Forecast':
        predict = nmc_json['data']['predict']
        ins = cls()
        ins.update_time = datetime.datetime.strptime(predict['publish_time'],
                                                     '%Y-%m-%d %H:%M')
        ins.city = nmc_json['data']['real']['station']['city']
        for day in predict['detail']:
            fc = ForecastDay(datetime.datetime.strptime(day['date'],
                                                        '%Y-%m-%d').date())
            high = int(day['day']['weather']['temperature'])
            low = int(day['night']['weather']['temperature'])
            fc.high = max(high, low)
            fc.low = min(high, low)
            fc.day_weather = day['day']['weather']['info']
            fc.night_weather = day['night']['weather']['info']
            ins.data.append(fc)
        return ins
            
    @classmethod
    def from_cache(cls, path: str = 'cache.json') -> 'Forecast':
        with open(path, 'r', encoding='utf8') as fin:
            return cls.from_nmc_json(json.load(fin))


class ForecastDay:
    
    def __init__(self, date: datetime.date = None) -> None:
        self.date = date
        self.high = None # type: Optional[int]
        self.low = None # type: Optional[int]
        self.day_weather = None # type: Optional[str]
        self.night_weather = None # type: Optional[str]


class ForecastPlot:
    
    LEFT = 0.2
    RIGHT = 0.95
    UP = 0.92
    BOTTOM = 0.04
    CITY_PADDING = 0.04
    TEMP_BUFFER = 0.25
    
    def __init__(self, forecasts: List[Forecast]) -> None:
        self.forecasts = forecasts
        self.dates = [fd.date for fd in forecasts[0].data[1:]]
        self.maxtemp = max(fd.high
                           for fc in self.forecasts
                           for fd in fc.data
                           if fd.high < 100) + self.TEMP_BUFFER
        self.mintemp = min(fd.low
                           for fc in self.forecasts
                           for fd in fc.data
                           if fd.high < 100) - self.TEMP_BUFFER
       
    @classmethod 
    def from_nmc(cls, codes: List[str]) -> 'ForecastPlot':
        return cls([Forecast.from_nmc(c) for c in codes])
    
    def plot(self):
        self.setup_plot()
        self.plot_date()
        self.save()
    
    def setup_plot(self):
        figsize = (5, 8)
        self.height = figsize[1] / figsize[0]
        self.fig = plt.figure(figsize=figsize, dpi=150)
        self.ax = plt.axes([0, 0, 1, 1])
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, self.height])
        self.ax.axis('off')
        
    def plot_date(self):
        for i, date in enumerate(self.dates):
            day_width_with_padding = (self.RIGHT - self.LEFT) / len(self.dates)
            DAY_WIDTH = day_width_with_padding * 0.8
            DAY_PADDING = day_width_with_padding * 0.2
            start = self.LEFT + i * (DAY_WIDTH + DAY_PADDING)
            end = start + DAY_WIDTH
            mid = (start + end) / 2
            
            # 画日期
            plt.text(mid, self.height * 0.95, date.strftime('%m/%d'), size=9,
                     ha='center', va='center', family='Arial', color='w')
            
            for j, fcst in enumerate(self.forecasts):
                CITY_HEIGHT = (self.UP - self.BOTTOM) * self.height / \
                    len(self.forecasts) - self.CITY_PADDING
                high = self.UP * self.height - j * (CITY_HEIGHT + \
                    self.CITY_PADDING)
                low = high - CITY_HEIGHT
                
                # 画浅色圆柱为底
                rr = RoundedRect(start, low, DAY_WIDTH, CITY_HEIGHT,
                                 radius=DAY_WIDTH * 0.4, facecolor='w',
                                 transform=self.ax.transData, alpha=0.03)
                self.ax.add_patch(rr.to_patch())
                
                fd = fcst.data[i+1]
                
                # 画天气符号
                wi = WeatherIcon(text=fd.day_weather, size=DAY_WIDTH * 0.4,
                                 x=mid, y=high - DAY_WIDTH * 0.3,
                                 facecolor='w', transform=self.ax.transData,
                                 edgecolor='none', alpha=0.8)
                self.ax.add_patch(wi.to_patch())
                
                wi = WeatherIcon(text=fd.night_weather, night=True,
                                 size=DAY_WIDTH * 0.4,
                                 x=mid, y=low + DAY_WIDTH * 0.3,
                                 facecolor='w', transform=self.ax.transData,
                                 edgecolor='none', alpha=0.8)
                self.ax.add_patch(wi.to_patch())
                
                # 画温度柱
                temp_space = CITY_HEIGHT * 0.1
                temp_max_high = high - DAY_WIDTH * 0.6 - temp_space
                temp_min_low = low + DAY_WIDTH * 0.6 + temp_space
                
                temp_high = (fd.high + self.TEMP_BUFFER - self.mintemp) / \
                    (self.maxtemp - self.mintemp) * \
                    (temp_max_high - temp_min_low) + temp_min_low
                temp_low = (fd.low - self.TEMP_BUFFER - self.mintemp) / \
                    (self.maxtemp - self.mintemp) * \
                    (temp_max_high - temp_min_low) + temp_min_low
                temp_width = DAY_WIDTH * 0.06
                
                tr = RoundedRect(mid - temp_width / 2,
                                 temp_low - temp_width / 2,
                                 temp_width, temp_high - temp_low,
                                 radius='full', facecolor='#F0AE06',
                                 transform=self.ax.transData)
                self.ax.add_patch(tr.to_patch())
                
                # 画高低气温
                plt.text(mid, temp_high + temp_space / 2, str(fd.high),
                         color='w', family='Arial', size=8, alpha=0.8,
                         va='center', ha='center')
                plt.text(mid, temp_low - temp_space / 1.5, str(fd.low),
                         color='w', family='Arial', size=8, alpha=0.8,
                         va='center', ha='center')
                
                if not i == 0:
                    continue
                
                # 画城市名
                plt.text(0.1, (high + low) / 2, '\n'.join(fcst.city), color='w',
                         size=16, family='Coca-Cola Care Font', ha='center',
                         va='center')
        
        # 画底层说明字体
        update_time = self.forecasts[0].update_time.strftime('%Y/%m/%d %H:%M')
        plt.text(0.5, 0.025, f'中央气象台城市天气预报   {update_time}   @中国气象爱好者 制图',
                 color='w', size=6, family='Source Han Sans CN', ha='center',
                 va='center', alpha=0.7)

    def save(self):
        plt.savefig('R.png', facecolor='#1F3461')
        

class RoundedRect:
    
    MAGIC = 0.552284749831
    # Refer to: https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves

    def __init__(self, llx, lly, width, height, radius=None, text=None,
                 facecolor=None, edgecolor=None, transform=None, aspect=None,
                 **kwargs):
        self.llx = llx
        self.lly = lly
        self.width = width
        self.height = height
        if radius is None:
            self.radius = 0
        elif radius == 'full':
            self.radius = min(width, height) / 2
        else:
            self.radius = radius
        self.radius = min(min(width, height) / 2, self.radius)
        self.text = text
        self.facecolor = facecolor or 'none'
        self.edgecolor = edgecolor or 'none'
        self.transform = transform
        self.aspect = aspect
        self.kwargs = kwargs

    def to_path(self) -> mpath.Path:
        x = self.llx
        y = self.lly
        r = self.radius
        rm = self.radius * RoundedRect.MAGIC
        w = self.width
        h = self.height
        vert_and_code = [
            ((x + r, y), mpath.Path.MOVETO),
            ((x + w - r, y), mpath.Path.LINETO),
            ((x + w - r + rm, y), mpath.Path.CURVE4),
            ((x + w, y + r - rm), mpath.Path.CURVE4),
            ((x + w, y + r), mpath.Path.CURVE4),
            ((x + w, y + h - r), mpath.Path.LINETO),
            ((x + w, y + h - r + rm), mpath.Path.CURVE4),
            ((x + w - r + rm, y + h), mpath.Path.CURVE4),
            ((x + w - r, y + h), mpath.Path.CURVE4),
            ((x + r, y + h), mpath.Path.LINETO),
            ((x + r - rm, y + h), mpath.Path.CURVE4),
            ((x, y + h - r + rm), mpath.Path.CURVE4),
            ((x, y + h - r), mpath.Path.CURVE4),
            ((x, y + r), mpath.Path.LINETO),
            ((x, y + r - rm), mpath.Path.CURVE4),
            ((x + r - rm, y), mpath.Path.CURVE4),
            ((x + r, y), mpath.Path.CURVE4)
        ]
        return mpath.Path(*zip(*vert_and_code))
    
    def to_patch(self) -> mpatch.PathPatch:
        return mpatch.PathPatch(self.to_path(), fc=self.facecolor,
                                ec=self.edgecolor, transform=self.transform,
                                **self.kwargs)
        

class WeatherIcon:
    
    PATH_CACHE = {} # type: Dict[str, mpath.Path]
    TEXT_TO_ICON_CODE = {
        '阴': '104',
        '小雨': '305',
        '中雨': '306',
        '大雨': '307',
        '暴雨': '310',
        '多云': ('101', '151'),
        '晴': ('100', '150'),
        '阵雨': ('300', '350'),
        '雨夹雪': '404'
    } # type: Dict[str, Union[str, Tuple[str, str]]]
    ORIGINAL_SIZE = 16
    
    def __init__(self, text: str, night: bool = False, facecolor: str = None,
                 edgecolor: str = None, transform=None, x: float = 0.,
                 y: float = 0., size: float = 0.1, **kwargs) -> None:
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.transform = transform
        self.x = x
        self.y = y
        self.size = size
        self.kwargs = kwargs
        self.text = text
        self.night = night
        
    def to_path(self) -> mpath.Path:
        return self.load_icon(self.text_to_icon_code(self.text, self.night))
        
    def to_patch(self):
        affine = mtrans.Affine2D().scale(-self.size / self.ORIGINAL_SIZE)\
            .translate(self.x + self.size / 2, self.y + self.size / 2)
        return mpatch.PathPatch(self.to_path(), fc=self.facecolor,
                                ec=self.edgecolor,
                                transform=affine + self.transform,
                                **self.kwargs)
    
    @classmethod
    def text_to_icon_code(cls, text: str, night: bool = False):
        if '到' in text:
            text = text[text.rfind('到')+1:]
        code = cls.TEXT_TO_ICON_CODE[text]
        if isinstance(code, str):
            return code
        if night:
            return code[1]
        return code[0]
    
    @classmethod
    def load_icon(cls, code: str) -> mpath.Path:
        return cls.PATH_CACHE.setdefault(code, parse_file(f'icons/{code}.svg'))
    


if __name__ == '__main__':
    codes = ['59287', '58847', '56778', '59493']
    fcst = [Forecast.from_nmc(code) for code in codes]
    fp = ForecastPlot(fcst)
    #fcst = Forecast.from_cache()
    #fp = ForecastPlot([fcst, fcst, fcst])
    fp.plot()
