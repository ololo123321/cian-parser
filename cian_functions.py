import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from stem import Signal
from stem.control import Controller
from fake_useragent import UserAgent as ua
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from functools import partial


TOR_PROXIES = {'http': 'socks5://localhost:9050',
               'https': 'socks5://localhost:9050'}

### Парсинг инфы о хатах из контента


def get_coords(soup):
    try:
        text = soup.find('script', type="text/javascript", src='').get_text()
        res = re.findall('"coordinates":{("\w+"):(\d+.\d+),("\w+"):(\d+.\d+)}', text)[0]
        d = {res[0].replace('"',''): res[1], res[2].replace('"',''): res[3]}
        return d['lat'], d['lng']
    except:
        return np.nan, np.nan


def get_dist_to_subway(soup):
    try:
        keys = [tag.get_text() for tag in soup.find_all('a', class_='underground_link--1qgA6')]
        values = [tag.get_text() for tag in soup.find_all('span', class_='underground_time--3SZFY')]
        if all(['пешком' in v for v in values]) or all(['транспорте' in v for v in values]):
            l = [int(re.sub('\D+','',v)) for v in values]
            min_time = min(l)
            nearest_station = keys[l.index(min_time)]
            if all(['пешком' in v for v in values]):
                way = 'пешком'
            else:
                way = 'транспорт'
        else:
            values = [v for v in values if 'пешком' in v]
            indices = [values.index(v) for v in values]
            keys = [keys[i] for i in indices]
            l = [int(re.sub('\D+','',v)) for v in values]
            min_time = min(l)
            nearest_station = keys[l.index(min_time)]
            way = 'пешком'
    except:
        try:
            nearest_station = soup.find('a', class_='underground_link--1qgA6').get_text()
        except:
            nearest_station = np.nan
        # мниимальное время и способ могут быть не указаны
        min_time = np.nan
        way = 'пешком'
    return nearest_station, min_time, way


def get_address(soup):
    try:
        tags = [tag.get_text() for tag in soup.find_all('a', class_='link--36RM5 address-item--1jDfG')]
        okrug = [tag for tag in tags if 'АО' in tag][0]
        try:  # может быть не указан
            district_patterns = ['р-н', 'поселение']
            district = [tag for tag in tags if re.search('|'.join(district_patterns), tag)][0]
        except:
            district = np.nan
        try:  # может быть не указан
            street_patterns = ['ул.', 'пер.', 'наб.', 'просп.', 'проезд', 'ш.', 'бул.']
            street = [tag for tag in tags if re.search('|'.join(street_patterns), tag)][0]
        except:
            street = np.nan
        house = tags[-1]
        return okrug, district, street, house
    except:
        return np.nan, np.nan, np.nan, np.nan


def get_info_from_title(soup):
    try:
        keys = [tag.get_text() for tag in soup.find_all('div', class_=re.compile('info-title--35hA6'))]
        values = [tag.get_text() for tag in soup.find_all('div', class_=re.compile('info-text--3GGPV'))]
        return {k: v for k,v in zip(keys, values) if k != 'Этаж'}
    except:
        return {}


def get_common_info1(soup):
    try:
        tags = soup.find('ul', class_=re.compile('container--2PoDX')).find_all('li')
        return [tag.get_text() for tag in tags]
    except:
        return []


def get_common_info2(soup):
    try:
        keys = [tag.get_text() for tag in soup.find_all('span', class_='name--2EeYd')]
        values = [tag.get_text() for tag in soup.find_all('span', class_='value--14qS3')]
        return dict(zip(keys, values))
    except:
        return {}


def get_price(soup):
    try:
        return soup.find('span', class_='price_value--XlUfS').get_text()
    except:
        return np.nan


def get_house_info(soup):
    try:
        keys = [tag.get_text() for tag in soup.find_all('div', class_='name--5JCnH')]
        values = [tag.get_text() for tag in soup.find_all('div', class_='value--1gdmq')]
        keys = ['Ставка аренды' if 'аренды' in k else k for k in keys]
        keys = [k + ' [дом]' if k in ['Динамика ставки за год ', 'Динамика цены за м2 за год ',
                                      'Средняя цена за м2', 'Ставка аренды'] else k for k in keys]
        return {k: v for k,v in zip(keys, values) if k != 'Этажнось'}
    except:
        return {}


def get_district_info(soup):
    try:
        keys = [tag.get_text() for tag in soup.find_all('div', class_='name--1eMFr')]
        values = [tag.get_text() for tag in soup.find_all('div', class_='value--2Bhwv')]
        keys = ['Средняя цена' if 'цена' in k and 'м2' not in k else k for k in keys]
        keys = ['Ставка аренды' if 'аренды' in k else k for k in keys]
        keys = [k + ' [район]' if k in ['Динамика ставки за год ', 'Динамика цены за м2 за год ',
                                        'Средняя цена за м2', 'Ставка аренды'] else k for k in keys]
        return dict(zip(keys, values))
    except:
        return {}


### Парсинг сырого контента
"""
Алгоритм (взято отсюда: https://stackoverflow.com/questions/30286293/make-requests-using-python-over-tor):
1. tor --hash-password mypassword
2. скопировать пароль
3. sudo -i gedit /etc/tor/torrc
4. раскомментить ControlPort 9051
5. раскомментировать HashedControlPassword
6. заменить пароль на скопированный
7. сохранить изменения в файле
8. sudo service tor restart
"""


def get_ip():
    """
    Данная функция не используется, но пусть она хоть тут будет
    """
    return re.sub('[^0-9\.]', '', requests.get("http://httpbin.org/ip", proxies=TOR_PROXIES).text)


def renew_connection():
    with Controller.from_port(port = 9051) as controller:  # ControlPort 9051 в файле torrc
        controller.authenticate(password="mypassword")  # tor --hash-password mypassword
        controller.signal(Signal.NEWNYM)


def get_links_from_page(soup):
    if soup.find('title').text == 'Страница не найдена — ЦИАН':
        print('page not found error')
        return set()
    try:
        tag = soup.find_all('script')[5]
        text = tag.get_text()
        return {'https://www.cian.ru/sale/flat/{}/'.format(url)
                for url in re.findall("\[([\d+,]+)\]", text)[-1].split(',')}
    except:
        print('links parsing error')
        return set()


def get_total_info_flat(soup):
    if soup.find('title').text == 'Страница не найдена — ЦИАН':
        print('page not found error')
        return {}

    labels = ['coords', 'dist_to_subway', 'address',
              'info_from_title', 'common_info1', 'common_info2',
              'house_info', 'district_info', 'price']
    funcs = [get_coords, get_dist_to_subway, get_address,
             get_info_from_title, get_common_info1, get_common_info2,
             get_house_info, get_district_info, get_price]

    res = {}
    for label, func in zip(labels, funcs):
        if label == 'coords':
            res['latitude'], res['longitude'] = func(soup)
        elif label == 'dist_to_subway':
            res['subway'], res['time'], res['way'] = func(soup)
        elif label == 'address':
            res['okrug'], res['district'], res['street'], res['house'] = func(soup)
        else:
            res[label] = func(soup)
    return res


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """
    На случай ConnectionError
    взято отсюда: https://www.peterbe.com/plog/best-practice-with-retries-with-requests
    """
    s = requests.Session()
    retry = Retry(total=retries,
                  read=retries,
                  connect=retries,
                  backoff_factor=backoff_factor,
                  status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


class CaptchaError(Exception):
    pass


def parse_url(url, parse_links=True):
    browser = ua().random
    s = requests_retry_session()
    k = 1
    attempt = 1
    res = None
    while res is None:
        try:
            time.sleep(abs(np.random.normal(2)))
            try:
                content = s.get(url, proxies=TOR_PROXIES, headers={'User-Agent': browser}).content
            except Exception as x:
                print('parsing content error: {}'.format(x.__class__.__name__))
                raise
            soup = BeautifulSoup(content, 'lxml')
            if soup.find('title').text == 'Captcha - база объявлений ЦИАН':
                raise CaptchaError
            else:  # это всегда должно исполняться
                if parse_links:
                    res = get_links_from_page(soup)
                else:
                    res = get_total_info_flat(soup)
        except CaptchaError:
            attempt += 1
            browser = ua().random  # смена браузера
            renew_connection()  # смена ip
            time.sleep(9)  # чтоб успел смениться ip
            if attempt % 10 == 1:  # если долго не выходит избежать каптчу, то делаем паузу...
                print('{} min waiting...'.format(k))
                time.sleep(k * 60)
                k += 3  # ...и увеличиваем её продолжительность, если совсем долго не ничего не получается
        except:  # если не получается спарсить контент (вроде, ни разу не вылезло)
            if parse_links:
                res = set()
            else:
                res = {}
    return res


def parse_cian(urls, parse_links=True, verbose=10):
    if parse_links:
        result = set()
    else:
        result = {}
    f = partial(parse_url, parse_links=parse_links)
    try:
        with ThreadPool() as pool:
            for i, res in enumerate(pool.imap(f, urls)):
                if i%verbose == 0:
                    print(i)
                if parse_links:
                    result |= res
                else:
                    result[urls[i]] = res
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        # если добавить pool.close(); pool.join(), то result не вернётся
        return result
    return result


### Создание матрицы X


def dict_to_df(result):
    dict_labels = {'common_info2': ['Высота потолков',
                                    'Этажей в доме',
                                    'Совмещённый санузел',
                                    'Площадь комнат+ обозначение смежных комнат- обозначение изолированных комнат',
                                    'Тип комнаты',
                                    'Тип жилья',
                                    'Статус',
                                    'Раздельный санузел',
                                    'Вид из окон',
                                    'Этаж',
                                    'Количество комнат',
                                    'Ремонт',
                                    'Отделка',
                                    'Тип дома'],
                   'district_info': ['Название',
                                     'Средняя цена за м2 [район]',
                                     'Средний возраст домов',
                                     'Регион',
                                     'Динамика ставки за год  [район]',
                                     'Динамика цены за м2 за год  [район]',
                                     'Ставка аренды [район]',
                                     'Население',
                                     'Средняя цена'],
                   'house_info': ['Динамика цены за м2 за год  [дом]',
                                  'Материалы стен',
                                  'Динамика ставки за год  [дом]',
                                  'Конструктив и состояние',
                                  'Квартиры и планировки',
                                  'Ставка аренды [дом]',
                                  'Подъездов',
                                  'Год постройки',
                                  'Средняя цена за м2 [дом]',
                                  'Квартир',
                                  'Аварийный'],
                   'info_from_title': ['Общая', 'Срок сдачи', 'Жилая', 'Построен', 'Кухня']}

    common_info1_values = ['Пассажирский лифт', 'Балкон', 'Грузовой лифт',
                           'Мусоропровод', 'Телефон', 'Паркинг', 'Лоджия']

    X = defaultdict(list)
    for link, info in result.items():
        X['link'].append(link)  # на всякий
        for label, desc in info.items():
            if label in dict_labels:
                for k in dict_labels[label]:
                    try:
                        X[k].append(desc[k])
                    except:
                        X[k].append(np.nan)
            elif label == 'common_info1':
                for v in common_info1_values:
                    X[v].append((v in desc) * 1)
            else:
                X[label].append(desc)
    X = pd.DataFrame(X)
    return X


def clean_numeric(text):
    try:
        return float(re.sub('[^\d+.]', '', text.replace(',', '.')))
    except:
        return np.nan


def expand_release_date(text):
    try:
        year = re.search('\d{4}', text).group(0)
        try:
            kvartal = re.findall('(\d) кв.', text)[0]
            return year, kvartal
        except:
            return year, np.nan
    except:
        return np.nan, np.nan


def make_X(result):
    X = dict_to_df(result)

    rename_dict = {'Аварийный': 'dangerous',
                   'Балкон': 'balkon',
                   'Вид из окон': 'view',
                   'Высота потолков': 'height',
                   'Год постройки': 'building_year',
                   'Грузовой лифт': 'elevator_gruz',
                   'Динамика ставки за год  [дом]': 'house_price_rate',
                   'Динамика ставки за год  [район]': 'district_price_rate',
                   'Динамика цены за м2 за год  [дом]': 'house_price_rate_m2',
                   'Динамика цены за м2 за год  [район]': 'district_price_rate_m2',
                   'Жилая': 'space_living',
                   'Квартир': 'space_rooms',
                   'Количество комнат': 'n_rooms',
                   'Кухня': 'space_kitchen',
                   'Лоджия': 'is_lodjia',
                   'Материалы стен': 'walls_material',
                   'Мусоропровод': 'is_musoroprovod',
                   'Население': 'population',
                   'Общая': 'space_total',
                   'Отделка': 'otdelka',
                   'Паркинг': 'parking',
                   'Пассажирский лифт': 'elevator_pass',
                   'Площадь комнат+ обозначение смежных комнат- обозначение изолированных комнат': 'space_rooms_expanded',
                   'Подъездов': 'n_podezdov',
                   'Раздельный санузел': 'n_sanuz_razdel',
                   'Ремонт': 'remont',
                   'Совмещённый санузел': 'n_sanuz_sovmes',
                   'Средний возраст домов': 'building_age_mean',
                   'Средняя цена': 'price_district_mean',
                   'Средняя цена за м2 [дом]': 'price_m2_house_mean',
                   'Средняя цена за м2 [район]': 'price_m2_house_district',
                   'Срок сдачи': 'release_date',
                   'Ставка аренды [дом]': 'rent_rate_house',
                   'Ставка аренды [район]': 'rent_rate_district',
                   'Статус': 'status',
                   'Телефон': 'is_phone',
                   'Тип дома': 'house_type',
                   'Тип жилья': 'zhilia_type',
                   'Тип комнаты': 'room_type',
                   'Этаж': 'floor',
                   'Этажей в доме': 'total_floors'}

    cols_to_drop = ['Квартиры и планировки', 'Конструктив и состояние', 'Название', 'Построен', 'Регион']

    X.rename(columns=rename_dict, inplace=True)
    X.drop(cols_to_drop, axis=1, inplace=True)

    num_cols = ['latitude', 'longitude', 'price', 'height', 'building_year',
                'house_price_rate', 'district_price_rate', 'house_price_rate_m2',
                'district_price_rate_m2', 'space_living', 'space_rooms',
                'n_rooms', 'space_kitchen', 'population', 'space_total',
                'n_podezdov', 'n_sanuz_razdel', 'n_sanuz_sovmes', 'building_age_mean',
                'price_district_mean', 'price_m2_house_mean', 'price_m2_house_district',
                'rent_rate_house', 'rent_rate_district', 'floor', 'total_floors']

    X[num_cols] = X[num_cols].applymap(clean_numeric)

    X.loc[X['district'].notnull(), 'district'] = \
        X.loc[X['district'].notnull(), 'district'].apply(lambda x: re.sub('р-н\s', '', x.lower()))
    X.loc[X['house'].notnull(), 'house'] = \
        X.loc[X['house'].notnull(), 'house'].apply(lambda x: x.lower() if re.sub('\D+', '', x).isdigit() else np.nan)

    street_patterns = ['ул.', 'пер.', 'наб.', 'просп.', 'проезд', 'ш.', 'бул.']
    X.loc[X['street'].notnull(), 'street'] = \
        X.loc[X['street'].notnull(), 'street'].apply(lambda x: re.sub('|'.join(street_patterns), '', x.lower()).strip())

    X['okrug'] = X['okrug'].str.lower()
    X['subway'] = X['subway'].str.lower()
    X['view'] = X['view'].str.lower()
    X['remont'] = X['remont'].str.lower()
    X['house_type'] = X['house_type'].str.lower()
    X['zhilia_type'] = X['zhilia_type'].str.lower()
    X['room_type'] = X['room_type'].str.lower()

    X['dangerous'] = X['dangerous'].replace('-', 'Нет').fillna('Нет').str.lower()
    X['walls_material'] = X['walls_material'].replace('-', np.nan).str.lower()

    X['release_year'], X['release_kvartal'] = zip(*X['release_date'].apply(expand_release_date))
    del X['release_date']

    return X