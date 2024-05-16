# strategy_of_individual.py
import plotly.graph_objects as go
import pandas_ta as ta
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots


def calculate_trades(df):
    df['trade'] = ''
    df.loc[df['Signal'] == 1, 'trade'] = 'buy'
    df.loc[df['Signal'] == -1, 'trade'] = 'sell'

    df['position'] = df['trade'].replace('', np.nan).ffill().fillna('')
    changes = df['position'].ne(df['position'].shift())
    df['start_trade'] = changes & df['position'].ne('')
    df['end_trade'] = changes.shift(-1).fillna(False) & df['position'].ne('')

    return df


def performance(df, start_date, end_date, capital, fee, slippage):
    df = calculate_trades(df)
    df_performance = df.copy()
    df_performance['Capital'] = capital
    df_performance['Strategy_Capital'] = capital
    df_performance['Returns'] = df['Close'].pct_change()  # 일일 수익률 계산
    df_performance['Trade_Returns'] = np.where(df_performance['end_trade'], df_performance['Returns'], np.nan)

    previous_signal = df_performance['Signal'].iloc[0]

    # 자본 계산
    for i in range(1, len(df_performance)):
        df_performance.loc[i, 'Capital'] = df_performance.loc[i - 1, 'Capital'] * (1 + df_performance.loc[i, 'Returns'])
        df_performance.loc[i, 'Capital'] = int(df_performance.loc[i, 'Capital'])  # Capital을 정수로 변환

        if df_performance.loc[i, 'Signal'] != previous_signal:
            trade_multiplier = 1 + df_performance.loc[i, 'Strategy_Returns'] - (fee + slippage)
            previous_signal = df_performance.loc[i, 'Signal']
        else:
            trade_multiplier = 1 + df_performance.loc[i, 'Strategy_Returns']

        df_performance.loc[i, 'Strategy_Capital'] *= trade_multiplier
    df_performance['Strategy_Capital'] = df_performance['Strategy_Capital'].astype(int)

    # 성과 지표 계산
    total_trades = df_performance['start_trade'].sum()
    successful_trades = df_performance[df_performance['Trade_Returns'] > 0]['end_trade'].sum()
    success_rate = successful_trades / total_trades if total_trades > 0 else 0
    average_trade_return = df_performance.loc[df_performance['end_trade'], 'Trade_Returns'].mean()

    df_performance['Total Trades'] = total_trades
    df_performance['Successful Trades'] = successful_trades
    df_performance['Success Rate'] = success_rate * 100  # 퍼센트로 표현
    df_performance['Average Trade Return'] = average_trade_return

    # 상관성 계산
    correlation = df_performance['Returns'].corr(df_performance['KOSPI'])
    df_performance['Correlation'] = correlation

    # 연평균 복리 수익률 (CAGR)
    total_days = (df_performance['Date'].iloc[-1] - df_performance['Date'].iloc[0]).days
    years = total_days / 365.25
    CAGR = ((df_performance['Strategy_Capital'].iloc[-1] / df_performance['Strategy_Capital'].iloc[0]) ** (1 / years)) - 1
    df_performance['CAGR'] = CAGR

    # 최대 낙폭 (MDD)
    peak = df_performance['Strategy_Capital'].cummax()
    drawdown = (df_performance['Strategy_Capital'] / peak) - 1
    max_dd = drawdown.min()
    df_performance['MDD'] = max_dd

    # 변동성
    vol = df_performance['Returns'].std() * np.sqrt(252)
    df_performance['Volatility'] = vol

    # 샤프 지수
    sharp_ratio = (df_performance['Returns'].mean()- 0.03) / df_performance['Returns'].std() * np.sqrt(252)
    df_performance['Sharpe Ratio'] = sharp_ratio

    return df_performance


def buy_and_hold_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 1  # 처음부터 끝까지 보유
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date  # 시간 정보 제거

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)  # 성능 계산

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='누적 수익률', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='바이 앤 홀드',
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Buy and Hold Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )
    columns = ['Date', 'Close', 'Signal', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
               'Cumulative_Strategy_Returns', 'KOSPI']
    df = df[columns]

    trade_stat = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    perform_metrics = ['Correlation', 'CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    trade_stat = df_performance[trade_stat]
    perform_metrics = df_performance[perform_metrics]



    return fig, df, trade_stat, perform_metrics

def mean_reversion_strategy_rsi(df, start_date, end_date, capital, fee, slippage, rsi_low=30, rsi_high=70): # 평균 회귀 전략

    df['Signal'] = 0
    df.loc[df['RSI'] < rsi_low, 'Signal'] = 1  # 과매도 매수
    df.loc[df['RSI'] > rsi_high, 'Signal'] = -1  # 과매수 매도
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='mean_reversion_strategy_rsi',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Mean Reversion Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    rsi_y_low = np.full(len(df['Date']), rsi_low)
    rsi_y_high = np.full(len(df['Date']), rsi_high)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=rsi_y_low, mode='lines', name='low', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=rsi_y_high, mode='lines', name='high', line=dict(color='red')))

    fig1.update_layout(
        title='RSI',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['RSI_Signal'] = np.where(df['RSI'] < rsi_low, 'Buy', np.where(df['RSI'] > rsi_high, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'RSI', 'RSI_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']

    trade_stat = df_performance[trade_stat]
    perform_metrics = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']

    perform_metrics = df_performance[perform_metrics]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics



def mean_reversion_strategy_cci(df, start_date, end_date, capital, fee, slippage, cci_low=-100, cci_high=100):
    df['Signal'] = 0
    df.loc[df['CCI'] < cci_low, 'Signal'] = 1  # 과매도 매수
    df.loc[df['CCI'] > cci_high, 'Signal'] = -1  # 과매수 매도
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='mean_reversion_strategy_cci',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Mean Reversion Strategy (CCI) vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    cci_y_low = np.full(len(df['Date']), cci_low)
    cci_y_high = np.full(len(df['Date']), cci_high)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], mode='lines', name='CCI', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=cci_y_low, mode='lines', name='low', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=cci_y_high, mode='lines', name='high', line=dict(color='red')))

    fig1.update_layout(
        title='CCI',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['CCI_Signal'] = np.where(df['CCI'] < cci_low, 'Buy', np.where(df['CCI'] > cci_high, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'CCI', 'CCI_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def momentum_strategy_ma_cross(df, start_date, end_date, capital, fee, slippage): #모멘텀 SMA
    df['Signal'] = 0
    df.loc[df['SMA_20'] > df['SMA_60'], 'Signal'] = 1  # 매수
    df.loc[df['SMA_20'] < df['SMA_60'], 'Signal'] = -1  # 매도
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='momentum_strategy_ma_cross',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Momentum Strategy (MA Cross) vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='price', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], mode='lines', name='SMA_20', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_60'], mode='lines', name='SMA_60', line=dict(color='red')))

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['SMA_Signal'] = np.where(df['SMA_20'] > df['SMA_60'], 'Buy',
                                np.where(df['SMA_20'] < df['SMA_60'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'SMA_20', 'SMA_60', 'SMA_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics

def momentum_strategy(df, start_date, end_date, capital, fee, slippage, threshold=0):
    df['Signal'] = 0
    df.loc[df['Momentum'] > threshold, 'Signal'] = 1  # 매수
    df.loc[df['Momentum'] < threshold, 'Signal'] = -1  # 매도
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='momentum_strategy',
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Momentum Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    threshold_x = np.full(len(df['Date']), threshold)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Momentum'], mode='lines', name='Momentum', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=threshold_x, mode='lines', name='threshold', line=dict(color='red')))

    fig1.update_layout(
        title='momentum',
        xaxis_title='Date',
        yaxis_title='momentum',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Momentum_Signal'] = np.where(df['Momentum'] > threshold, 'Buy',
                                     np.where(df['Momentum'] < threshold, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'Momentum', 'Momentum_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics

def bollinger_band_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1  # 매수
    df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = -1  # 매도
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='bollinger_band_strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Bollinger Band Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='price', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='low', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='high', line=dict(color='red')))

    fig1.update_layout(
        title='Bollinger Band',
        xaxis_title='Date',
        yaxis_title='Bollinger Band',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 'Buy',
                               np.where(df['Close'] > df['BB_Upper'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'BB_Upper', 'BB_Mid', 'BB_Lower', 'BB_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def golden_cross_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0

    # 골든 크로스 조건을 기반으로 매수/매도 신호 설정
    df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1  # 매수
    df.loc[df['SMA_50'] < df['SMA_200'], 'Signal'] = 0  # 보유 및 매도

    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold', line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Cumulative_Strategy_Returns'], mode='lines', name='golden_cross_strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Golden Cross Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='price', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA_50', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], mode='lines', name='SMA_200', line=dict(color='red')))

    fig1.update_layout(
        title='GOLDEN CROSS',
        xaxis_title='Date',
        yaxis_title='PRICE',
        height=600
    )
    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Golden_Cross_Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 'Buy',
                                         np.where(df['SMA_50'] < df['SMA_200'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'SMA_50', 'SMA_200', 'Golden_Cross_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def dead_cross_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0

    df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = -1  # 매도
    df.loc[df['SMA_50'] < df['SMA_200'], 'Signal'] = 0  # 보유 및 매수

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy_Returns'], mode='lines', name='Dead_cross_strategy',
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Dead Cross Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='price', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], mode='lines', name='SMA_50', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], mode='lines', name='SMA_200', line=dict(color='red')))

    fig1.update_layout(
        title='Dead CROSS',
        xaxis_title='Date',
        yaxis_title='PRICE',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Dead_Cross_Signal'] = np.where(df['SMA_50'] > df['SMA_200'], 'Sell',
                                       np.where(df['SMA_50'] < df['SMA_200'], 'Buy', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'SMA_50', 'SMA_200', 'Dead_Cross_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics

def macd_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0

    # MACD 교차를 기준으로 매수/매도 신호 설정
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1  # 매수
    df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1  # 매도

    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='macd_strategy',
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='MACD Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='green')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['MACD_Signal'], mode='lines', name='MACD_Signal', line=dict(color='blue')))

    fig1.update_layout(
        title='MACD',
        xaxis_title='Date',
        yaxis_title='MACD',
        height=600
    )
    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['MACD_Signal'] = np.where(df['MACD'] > df['MACD_Signal'], 'Buy',
                                 np.where(df['MACD'] < df['MACD_Signal'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics

def envelope_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['Close'] > df['Upper'], 'Signal'] = -1  # 과매수에서 매도
    df.loc[df['Close'] < df['Lower'], 'Signal'] = 1   # 과매도에서 매수
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()
    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='Envelope Strategy', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(title='Envelope Strategy vs Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative_Returns', height=600)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['Upper'], mode='lines', name='Upper Envelope', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['MA'], mode='lines', name='Moving Average', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], mode='lines', name='Lower Envelope', line=dict(color='red')))

    fig1.update_layout(
        title='Moving Average Envelopes',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Envelope_Signal'] = np.where(df['Close'] < df['Lower'], 'Buy',
                                     np.where(df['Close'] > df['Upper'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'MA', 'Upper', 'Lower', 'Envelope_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics



def bull_bear_power_strategy(df, start_date, end_date, capital, fee, slippage, bull_threshold=5, bear_threshold=-5):
    df['Bull_Power'] = df['High'] - df['EMA_20']
    df['Bear_Power'] = df['Low'] - df['EMA_20']

    df['Signal'] = 0
    df.loc[df['Bear_Power'] < bear_threshold, 'Signal'] = 1  # 매수
    df.loc[df['Bull_Power'] > bull_threshold, 'Signal'] = -1  # 매도

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()
    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='bull_bear_power_strategy', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='price', line=dict(color='black')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['EMA_20'], mode='lines', name='EMA', line=dict(color='orange')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Bull_Power'], mode='lines', name='Bull_Power', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Bear_Power'], mode='lines', name='Bear_Power', line=dict(color='red')))


    fig.update_layout(title='Bull/Bear Power Strategy', xaxis_title='Date', yaxis_title='Cumulative_Returns', height=600)

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['BullBear_Signal'] = np.where(df['Bear_Power'] < bear_threshold, 'Buy',
                                     np.where(df['Bull_Power'] > bull_threshold, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'Bull_Power','EMA_20','Bear_Power', 'BullBear_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics



def trend_following_adx(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0

    df.loc[(df['ADX'] > 25) & (df['+DI'] > df['-DI']), 'Signal'] = 1  # 매수
    df.loc[(df['ADX'] > 25) & (df['+DI'] < df['-DI']), 'Signal'] = -1  # 매도

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='trend_following_adx',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Trend Following (ADX) vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['ADX'], mode='lines', name='ADX', line=dict(color='green')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['+DI'], mode='lines', name='+DI', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['-DI'], mode='lines', name='-DI', line=dict(color='blue')))

    fig1.update_layout(
        title='ADX',
        xaxis_title='Date',
        yaxis_title='ADX',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['ADX_Signal'] = np.where((df['ADX'] > 25) & (df['+DI'] > df['-DI']), 'Buy',
                                np.where((df['ADX'] > 25) & (df['+DI'] < df['-DI']), 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'ADX', '+DI', '-DI', 'ADX_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def breakout_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0

    df.loc[df['Close'] > df['High_Breakout'], 'Signal'] = 1  # 매수
    df.loc[df['Close'] < df['Low_Breakout'], 'Signal'] = -1  # 매도

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='mean_reversion_strategy_rsi',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Breakout Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='pirce', line=dict(color='green')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['High_Breakout'], mode='lines', name='High_Breakout', line=dict(color='red')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['Low_Breakout'], mode='lines', name='Low_Breakout', line=dict(color='blue')))

    fig1.update_layout(
        title='breakout',
        xaxis_title='Date',
        yaxis_title='Breakout',
        height=600
    )
    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Breakout_Signal'] = np.where(df['Close'] > df['High_Breakout'], 'Buy',
                                     np.where(df['Close'] < df['Low_Breakout'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'High_Breakout', 'Low_Breakout', 'Breakout_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def donchian_channel_strategy(df, start_date, end_date, capital, fee, slippage):
    """돈치안 채널 전략"""
    df['Signal'] = 0

    # 매수/매도 신호 설정
    df.loc[df['Close'] > df['High_Channel'], 'Signal'] = 1  # 매수
    df.loc[df['Close'] < df['Low_Channel'], 'Signal'] = -1  # 매도

    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='donchian_channel_strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Donchian Channel Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='RSI', line=dict(color='green')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['Low_Channel'], mode='lines', name='Low_Channel', line=dict(color='blue')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['High_Channel'], mode='lines', name='High_Channel', line=dict(color='red')))

    fig1.update_layout(
        title='donchain Channel',
        xaxis_title='Date',
        yaxis_title='donchain Channel',
        height=600
    )
    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Donchian_Signal'] = np.where(df['Close'] > df['High_Channel'], 'Buy',
                                     np.where(df['Close'] < df['Low_Channel'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'High_Channel', 'Low_Channel', 'Donchian_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def keltner_channel_strategy(df, start_date, end_date, capital, fee, slippage, atr_length=10, multiplier=2):
    """켈트너 채널 전략"""
    df['Signal'] = 0

    df['High_Channel'] = df['EMA_20'] + multiplier * df['ATR']
    df['Low_Channel'] = df['EMA_20'] - multiplier * df['ATR']

    # 매수/매도 신호 설정
    df.loc[df['Close'] > df['High_Channel'], 'Signal'] = 1  # 매수
    df.loc[df['Close'] < df['Low_Channel'], 'Signal'] = -1  # 매도

    # 수익률 계산
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()

    # 누적 수익률 계산
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='keltner_channel_strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Keltner Channel Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='RSI', line=dict(color='green')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['Low_Channel'], mode='lines', name='Low_Channel', line=dict(color='blue')))
    fig1.add_trace(
        go.Scatter(x=df['Date'], y=df['High_Channel'], mode='lines', name='High_Channel', line=dict(color='red')))

    fig1.update_layout(
        title='keltner Channel',
        xaxis_title='Date',
        yaxis_title='keltner Channel',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Keltner_Signal'] = np.where(df['Close'] > df['High_Channel'], 'Buy',
                                    np.where(df['Close'] < df['Low_Channel'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'High_Channel', 'Low_Channel', 'Keltner_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def wma_crossover_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['WMA_10'] > df['WMA_30'], 'Signal'] = 1
    df.loc[df['WMA_10'] < df['WMA_30'], 'Signal'] = -1

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='WMA Strategy', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='WMA Crossover Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['WMA_10'], mode='lines', name='WMA_10', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['WMA_30'], mode='lines', name='WMA_30', line=dict(color='red')))

    # 매수 및 매도 신호 추가
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig1.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal',
                              marker=dict(color='green', symbol='triangle-up', size=10)))
    fig1.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal',
                              marker=dict(color='red', symbol='triangle-down', size=10)))

    fig1.update_layout(
        title='WMA Crossover with Buy/Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['WMA_Crossover_Signal'] = np.where(df['WMA_10'] > df['WMA_30'], 'Buy', np.where(df['WMA_10'] < df['WMA_30'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'WMA_10', 'WMA_30', 'WMA_Crossover_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def stochastic_oscillator_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['%K'] > df['%D'], 'Signal'] = 1
    df.loc[df['%K'] < df['%D'], 'Signal'] = -1

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines',
                             name='Stochastic Oscillator Strategy', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Stochastic Oscillator Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # 첫 번째 서브플롯에 %K와 %D 추가
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['%K'], mode='lines', name='%K', line=dict(color='blue')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['%D'], mode='lines', name='%D', line=dict(color='red')), row=1, col=1)

    # 두 번째 서브플롯에 Close 추가
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')), row=2,
                   col=1)

    # 매수 및 매도 신호 추가 (두 번째 서브플롯에 추가)
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig1.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal',
                              marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal',
                              marker=dict(color='red', symbol='triangle-down', size=10)), row=2, col=1)

    # 레이아웃 업데이트
    fig1.update_layout(
        title='Stochastic Oscillator with Buy/Sell Signals',
        xaxis_title='Date',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Stochastic_Signal'] = np.where(df['%K'] > df['%D'], 'Buy', np.where(df['%K'] < df['%D'], 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', '%K', '%D', 'Stochastic_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def stochastic_rsi_strategy(df, start_date, end_date, capital, fee, slippage):
    df = df.copy()

    df['Signal'] = 0
    df.loc[df['STOCH_RSI'] < 20, 'Signal'] = 1
    df.loc[df['STOCH_RSI'] > 80, 'Signal'] = -1

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='Stochastic RSI Strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Stochastic RSI Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['STOCH_RSI'], mode='lines', name='Stoch RSI', line=dict(color='blue')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')), row=2, col=1)

    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig1.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal',
                              marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal',
                              marker=dict(color='red', symbol='triangle-down', size=10)), row=2, col=1)

    # 레이아웃 업데이트
    fig1.update_layout(
        title='Stochastic RSI with Buy/Sell Signals',
        xaxis_title='Date',
        height=600
    )


    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Stochastic_RSI_Signal'] = np.where(df['STOCH_RSI'] < 20, 'Buy', np.where(df['STOCH_RSI'] > 80, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'STOCH_RSI', 'Stochastic_RSI_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def williams_r_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['Williams_%R'] < -80, 'Signal'] = 1
    df.loc[df['Williams_%R'] > -20, 'Signal'] = -1

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='Williams %R Strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Williams %R Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)


    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Williams_%R'], mode='lines', name='Williams %R', line=dict(color='blue')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')), row=2, col=1)

    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig1.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)), row=2, col=1)

    fig1.update_layout(
        title='Williams %R with Buy/Sell Signals',
        xaxis_title='Date',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Williams_R_Signal'] = np.where(df['Williams_%R'] < -80, 'Buy',
                                       np.where(df['Williams_%R'] > -20, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'Williams_%R', 'Williams_R_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


def ultimate_oscillator_strategy(df, start_date, end_date, capital, fee, slippage):
    df['Signal'] = 0
    df.loc[df['Ultimate_Osc'] < 30, 'Signal'] = 1
    df.loc[df['Ultimate_Osc'] > 70, 'Signal'] = -1

    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift()
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

    kospi = fdr.DataReader('KS11', start_date, end_date)
    kospi['KOSPI_Returns'] = kospi['Close'].pct_change()
    kospi['Cumulative_KOSPI_Returns'] = (1 + kospi['KOSPI_Returns']).cumprod()

    df['KOSPI'] = kospi['Cumulative_KOSPI_Returns']

    df = df.reset_index()
    df['Date'] = df['Date'].dt.date

    df_performance = performance(df, start_date, end_date, capital, fee, slippage)

    # 그래프 그리기
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative_Returns'], mode='lines', name='Buy and Hold',
                             line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Cumulative_Strategy_Returns'], mode='lines', name='Ultimate Oscillator Strategy',
                   line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['KOSPI'], mode='lines', name='KOSPI', line=dict(color='green')))

    fig.update_layout(
        title='Ultimate Oscillator Strategy vs Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        height=600
    )

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # 첫 번째 서브플롯에 Ultimate Oscillator 추가
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Ultimate_Osc'], mode='lines', name='Ultimate Oscillator',
                              line=dict(color='blue')), row=1, col=1)

    # 두 번째 서브플롯에 Close 추가
    fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(color='black')), row=2,
                   col=1)

    # 매수 및 매도 신호 추가 (두 번째 서브플롯에 추가)
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    fig1.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal',
                              marker=dict(color='green', symbol='triangle-up', size=10)), row=2, col=1)
    fig1.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal',
                              marker=dict(color='red', symbol='triangle-down', size=10)), row=2, col=1)

    # 레이아웃 업데이트
    fig1.update_layout(
        title='Ultimate Oscillator with Buy/Sell Signals',
        xaxis_title='Date',
        height=600
    )

    df = pd.merge(df, df_performance, on='Date', suffixes=('', '_perf'))

    returns_c = ['Date', 'Close', 'Capital', 'Strategy_Capital', 'Returns', 'Strategy_Returns', 'Cumulative_Returns',
                 'Cumulative_Strategy_Returns', 'KOSPI']
    returns = df[returns_c]

    df['Ultimate_Oscillator_Signal'] = np.where(df['Ultimate_Osc'] < 30, 'Buy',
                                                np.where(df['Ultimate_Osc'] > 70, 'Sell', 'Hold'))
    df['Buy/Sell_Signal'] = df['Signal'].replace({1: 'Buy', -1: 'Sell', 0: 'Hold'})
    df_2 = df[['Date', 'Close', 'Ultimate_Osc', 'Ultimate_Oscillator_Signal', 'Buy/Sell_Signal']]

    trade_c = ['trade', 'start_trade', 'end_trade', 'Trade_Returns']
    trade_info = df[trade_c]

    trade_stat_c = ['Total Trades', 'Successful Trades', 'Success Rate', 'Average Trade Return']
    trade_stat = df_performance[trade_stat_c]

    perform_metrics_c = ['Correlation','CAGR', 'MDD', 'Volatility', 'Sharpe Ratio']
    perform_metrics = df_performance[perform_metrics_c]

    return fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics


STRATEGY_MAPPING = {
    0: buy_and_hold_strategy,
    1: mean_reversion_strategy_rsi,
    2: mean_reversion_strategy_cci,
    3: momentum_strategy_ma_cross,
    4: momentum_strategy,
    5: bollinger_band_strategy,
    6: golden_cross_strategy,
    7: dead_cross_strategy,
    8: macd_strategy,
    9: envelope_strategy,
    10: bull_bear_power_strategy,
    11: trend_following_adx,
    12: breakout_strategy,
    13: donchian_channel_strategy,
    14: keltner_channel_strategy,
    15: wma_crossover_strategy,
    16: stochastic_oscillator_strategy,
    17: stochastic_rsi_strategy,
    18: williams_r_strategy,
    19: ultimate_oscillator_strategy
}
