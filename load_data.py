# load_data.py
import requests as rq
from bs4 import BeautifulSoup
import re
from io import BytesIO
import pandas as pd
import pandas_ta as ta

def cal_indicators(df):
    df = df.copy()
    buffer_days = 300

    df_extended = pd.concat([pd.DataFrame(index=pd.date_range(start=df.index[0] - pd.Timedelta(days=buffer_days), end=df.index[0] - pd.Timedelta(days=1))), df])

    # SMA (20, 50, 60, 200) ㅇ
    df_extended.loc[:, 'SMA_20'] = ta.sma(df_extended['Close'], 20)
    df_extended.loc[:, 'SMA_50'] = ta.sma(df_extended['Close'], 50)
    df_extended.loc[:, 'SMA_60'] = ta.sma(df_extended['Close'], 60)
    df_extended.loc[:, 'SMA_200'] = ta.sma(df_extended['Close'], 200)

    # EMA (지수이동평균) ㅇ
    df_extended.loc[:, 'EMA_20'] = ta.ema(df_extended['Close'], 20)
    df_extended.loc[:, 'EMA_60'] = ta.ema(df_extended['Close'], 60)

    # WMA(가중평균) ㅇ
    df_extended.loc[:, 'WMA_10'] = ta.wma(df_extended['Close'], length=10)
    df_extended.loc[:, 'WMA_30'] = ta.wma(df_extended['Close'], length=30)

    # RSI (14일) ㅇ
    df_extended.loc[:, 'RSI'] = ta.rsi(df_extended['Close'], length=14)

    # Stochastic RSI
    df_extended.loc[:, 'STOCH_RSI'] = ta.stochrsi(df_extended['Close'])['STOCHRSIk_14_14_3_3']

    # Stochastic_Oscillator ㅇ
    stochastic = ta.stoch(df_extended['High'], df_extended['Low'], df_extended['Close'])
    df_extended['%K'], df_extended['%D'] = stochastic['STOCHk_14_3_3'], stochastic['STOCHd_14_3_3']

    # MACD ㅇ
    macd = ta.macd(df_extended['Close'])
    df_extended.loc[:, 'MACD'], df_extended.loc[:, 'MACD_Hist'], df_extended.loc[:, 'MACD_Signal'] = macd['MACD_12_26_9'], macd['MACDh_12_26_9'], macd['MACDs_12_26_9']

    # ADX ㅇ
    adx = ta.adx(df_extended['High'], df_extended['Low'], df_extended['Close'])
    df_extended.loc[:, 'ADX'], df_extended.loc[:, '+DI'], df_extended.loc[:, '-DI'] = adx['ADX_14'], adx['DMP_14'], adx['DMN_14']

    # Williams %R ㅇ
    df_extended.loc[:, 'Williams_R'] = ta.willr(df_extended['High'], df_extended['Low'], df_extended['Close'], length=14)

    # Commodity Channel Index ㅇ
    df_extended.loc[:, 'CCI'] = ta.cci(df_extended['High'], df_extended['Low'], df_extended['Close'])

    # Ultimate Oscillator ㅇ
    df_extended.loc[:, 'Ult_Osc'] = ta.uo(df_extended['High'], df_extended['Low'], df_extended['Close'])

    # Rate of Change
    df_extended.loc[:, 'ROC'] = ta.roc(df_extended['Close'])

    # Bull/Bear Power ㅇ
    df_extended.loc[:, 'Bull_Power'], df_extended.loc[:, 'Bear_Power'] = df_extended['High'] - df_extended['Close'], df_extended['Low'] - df_extended['Close']

    # Momentum ㅇ
    df_extended.loc[:, 'Momentum'] = ta.mom(df_extended['Close'], length=14)

    # Bollinger Bands ㅇ
    bb = ta.bbands(df_extended['Close'], length=20, std=2)
    df_extended.loc[:, 'BB_Upper'], df_extended.loc[:, 'BB_Mid'], df_extended.loc[:, 'BB_Lower'] = bb['BBU_20_2.0'], bb['BBM_20_2.0'], bb['BBL_20_2.0']

    # Envelope ㅇ
    df_extended['MA'] = df_extended['Close'].rolling(window=20).mean()
    df_extended['Upper'] = df_extended['MA'] * (1 + 0.05)
    df_extended['Lower'] = df_extended['MA'] * (1 - 0.05)

    # ATR ㅇ
    df_extended.loc[:, 'ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=10)

    #Williams_%R ㅇ
    df_extended.loc[:, 'Williams_%R'] = ta.willr(df['High'], df['Low'], df['Close'])

    #Ultimate_oscillator ㅇ
    df_extended.loc[:, 'Ultimate_Osc'] = ta.uo(df['High'], df['Low'], df['Close'])

    df_extended.loc[:,'High_Channel'] = df['High'].rolling(window=20).max()
    df_extended.loc[:, 'Low_Channel'] = df['Low'].rolling(window=20).min()

    df_extended.loc[:,'High_Breakout'] = df['High'].rolling(window=20).max()
    df_extended.loc[:,'Low_Breakout'] = df['Low'].rolling(window=20).min()

    df = df_extended.loc[df.index]

    return df

def load_data():
    url = 'https://finance.naver.com/sise/sise_deposit.nhn'
    data = rq.get(url)
    data_html = BeautifulSoup(data.content, 'html.parser')
    parse_day = data_html.select_one('div.subtop_sise_graph2 > ul.subtop_chart_note > li > span.tah').text
    biz_day = re.findall('[0-9]+', parse_day)
    biz_day = ''.join(biz_day)

    gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
    gen_otp_stk = {
        'mktId': 'STK',
        'trdDd': biz_day,
        'money': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
    }
    headers = {'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader'}
    otp_stk = rq.post(gen_otp_url, gen_otp_stk, headers=headers).text

    down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
    down_sector_stk = rq.post(down_url, {'code': otp_stk}, headers=headers)
    sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')

    df_section = pd.DataFrame(sector_stk)
    df_subset = df_section[['종목코드', '종목명', '업종명']]

    return pd.Series(df_subset['종목코드'].values, index=df_subset['종목명']).to_dict()


