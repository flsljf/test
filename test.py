#test.py
import streamlit as st
from datetime import datetime, timedelta
from load_data import load_data
from load_data import cal_indicators
from strategy_of_indivual import *

def data_setting(selected_company, start_date_str, end_date_str, data):
    stock_code = data.get(selected_company)
    if stock_code:
        buffer_days = 300  # 가장 긴 이동 평균 window 크기 이상으로 설정
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        buffer_start_date = (start_date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')

        price_data = fdr.DataReader(stock_code, buffer_start_date, end_date_str)
        price_data.index = pd.to_datetime(price_data.index)

        if start_date in price_data.index:
            start_idx = price_data.index.get_loc(start_date)
        else:
            closest_idx = price_data.index.get_indexer([start_date], method='pad')[0]
            start_idx = closest_idx

        buffer_idx = max(0, start_idx - buffer_days)
        eval_df = price_data.iloc[buffer_idx:]
        df_indicator = cal_indicators(eval_df)
        df_indicator = df_indicator[df_indicator.index >= start_date_str]

        return df_indicator
    return None


def main():
    st.title("Financial Strategy Analysis")

    # 사용자 입력 받기
    selected_company = st.selectbox("Select a company", options=list(load_data().keys()))
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    col3, col4, col5 = st.columns(3)
    with col3:
        capital = st.number_input("Capital", min_value=0.0, format='%f')
    with col4:
        fee = st.number_input("Fee (%)", min_value=0.0, format='%f')
    with col5:
        slippage = st.number_input("Slippage (%)", min_value=0.0, format='%f')

    # 전략 선택
    strategy_options = list(STRATEGY_MAPPING.keys())
    strategy_names = [
        "Buy and Hold", "Mean Reversion (RSI)", "Mean Reversion (CCI)",
        "Momentum (MA Cross)", "Momentum", "Bollinger Band", "Golden Cross",
        "Dead Cross", "MACD", "envelope_strategy", "bull_bear_power_strategy",
        "Trend Following (ADX)", "Breakout", "Donchian Channel", "Keltner Channel", "wma_crossover_strategy",
        "stochastic_oscillator_strategy", "stochastic_rsi_strategy", "Williams_R", "Ultimate_Oscillator"
    ]
    strategy_choice = st.selectbox("Choose a strategy", options=strategy_options,
                                   format_func=lambda x: strategy_names[x])

    if st.button("Run Strategy"):
        data = load_data()
        df = data_setting(selected_company, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), data)
        strategy_function = STRATEGY_MAPPING[strategy_choice]

        if strategy_choice == 0:
            fig, df, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Buy and Hold")
            st.plotly_chart(fig)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width = 1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width = 1000)
        elif strategy_choice == 1:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics  = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns)
            st.write("RSI GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 2:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date,end_date, capital,fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("CCI GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 3:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital,fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Momentum (ma-cross) GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 4:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Momentum GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 5:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Bollinger Band GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 6:  # golden cross
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Golden Cross GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 7:  # dead cross
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Dead Cross GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 8:  # MACD
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Dead Cross GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 9:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("envelope_strategy GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width= 1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 10:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("trend_following_adx GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 11:  # ADX
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("BULL / BEAR Power GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 12:  # Breakout
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("BreakOut GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 13:  # Donchian
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Donchian Channel GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 14:  # Keltner
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Keltner Channel GRAPH")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 15:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("wma_crossover Strategy Graph")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 16:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("stochastic_rsi_strategy Graph")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 17:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("stochastic_rsi_strategy Graph")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 18:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("Williams_R Graph")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        elif strategy_choice == 19:
            fig, fig1, returns, df_2, trade_info, trade_stat, perform_metrics = strategy_function(df, start_date, end_date, capital, fee, slippage)
            st.write("Returns")
            st.plotly_chart(fig)
            st.dataframe(returns, width=1500)
            st.write("ultimate_oscillator_strategy Graph")
            st.plotly_chart(fig1)
            st.dataframe(df_2, width=1000)
            st.write("Trading Statistics")
            st.dataframe(trade_stat.tail(1).reset_index(drop=True), width=1000)
            st.write("Performance Metrics")
            st.dataframe(perform_metrics.tail(1).reset_index(drop=True), width=1000)
        else:
            fig, strategy_df = strategy_function(df, start_date, end_date)
            st.write("Strategy Data:")
            st.plotly_chart(fig)
            st.dataframe(strategy_df.tail())

if __name__ == "__main__":
    main()
