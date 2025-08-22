import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import pandas as pd
import apimoex
import requests
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta

# Настройка единой цветовой палитры
COLOR_PRIMARY = "#D62728"
COLOR_SECONDARY = "#808080"
COLOR_BACKGROUND = "#FFFFFF"
COLOR_TEXT = "#4D4D4D"

st.set_page_config(layout="wide")
st.title("Симулятор инвестиций и анализа портфеля")
st.markdown("---")

# --- 1. Ввод данных пользователем ---

@st.cache_data(ttl=3600)
def get_securities():
    with requests.Session() as session:
        try:
            data = apimoex.get_board_securities(session, board='TQBR')
            df = pd.DataFrame(data)
            needed_cols = ['SECID', 'SHORTNAME', 'PREVPRICE', 'LOTSIZE', 'FACEVALUE', 'CURRENCYID', 'ISIN']
            existing_cols = [col for col in needed_cols if col in df.columns]
            df = df[existing_cols]
            return df
        except Exception as e:
            st.error(f"Ошибка при загрузке списка акций: {e}")
            return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_history(ticker, start, end):
    with requests.Session() as session:
        try:
            data = apimoex.get_market_candles(
                session, security=ticker, interval=24, start=start, end=end
            )
            df = pd.DataFrame(data)
            if not df.empty:
                df['begin'] = pd.to_datetime(df['begin'])
                df.set_index('begin', inplace=True)
            return df
        except Exception as e:
            st.warning(f"Ошибка при загрузке данных для тикера {ticker}: {e}")
            return pd.DataFrame()

securities_df = get_securities()

if not securities_df.empty:
    st.sidebar.header("Параметры симуляции")
    tickers = st.sidebar.multiselect(
        "Выберите акции для портфеля (до 5)",
        securities_df['SECID'],
        default=securities_df['SECID'][:2] if not securities_df.empty else []
    )

    weights = []
    st.sidebar.write("Задайте доли для каждой акции (сумма должна быть 1.0):")
    cols = st.sidebar.columns(len(tickers) if tickers else 1)
    
    if tickers:
        for i, ticker in enumerate(tickers):
            with cols[i]:
                weight = st.number_input(
                    f"{ticker}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(tickers) if tickers else 0.0,
                    step=0.01,
                    key=f"weight_{ticker}"
                )
                weights.append(weight)

    if tickers and (abs(sum(weights) - 1.0) > 0.01):
        st.sidebar.warning(f"Сумма долей должна быть равна 1.0. Текущая сумма: {sum(weights):.2f}")
        st.stop()

    start_date = st.sidebar.date_input("Дата начала анализа", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("Дата конца анализа", datetime.now())

    st.markdown("---")

    # --- 2. Основной блок дашборда ---

    if tickers:
        portfolio_df = pd.DataFrame()
        for ticker in tickers:
            df = get_history(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not df.empty:
                portfolio_df[ticker] = df['close']
        
        if portfolio_df.empty or len(portfolio_df) < 2:
            st.error("Недостаточно данных для выбранных акций или периода. Пожалуйста, измените выбор.")
            st.stop()
        
        imoex_df = get_history('IMOEX', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # --- Блок 1: Ключевые показатели портфеля ---
        st.header("1. Эффективность портфеля")
        st.divider()
        st.write("Здесь вы можете увидеть, насколько эффективен ваш портфель с точки зрения доходности и риска.")

        returns = portfolio_df.pct_change().dropna()
        if returns.empty:
            st.warning("Недостаточно данных для расчета доходности.")
            st.stop()

        weighted_returns = returns.dot(weights)
        total_return = (1 + weighted_returns).prod() - 1
        risk = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = total_return / risk if risk != 0 else 0
        
        col1, col2, col3 = st.columns(3, gap="large")
        col1.metric("Доходность", f"{total_return:.2%}", help="Общий прирост стоимости портфеля за период.")
        col2.metric("Риск (Волатильность)", f"{risk:.2%}", help="Показатель колебаний стоимости портфеля.")
        col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}", help="Доходность на единицу риска. Чем выше, тем лучше.")
        
        # --- Блок 2: Динамика и сравнение с рынком ---
        st.header("2. Динамика и сравнение с индексом")
        st.divider()
        st.write("График показывает, как стоимость вашего портфеля изменялась во времени по сравнению с рыночным индексом IMOEX.")
        
        cumulative = (1 + weighted_returns).cumprod()
        fig_cum = go.Figure(layout=go.Layout(template="plotly_white"))
        fig_cum.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name="Портфель", line=dict(color=COLOR_PRIMARY)))
        
        if not imoex_df.empty:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            imoex_cum = (1 + imoex_returns).cumprod()
            fig_cum.add_trace(go.Scatter(x=imoex_cum.index, y=imoex_cum, name="Индекс Мосбиржи (IMOEX)", line=dict(color=COLOR_SECONDARY, dash='dash')))

        fig_cum.update_layout(
            title="Сравнение кумулятивной доходности",
            xaxis_title="Дата",
            yaxis_title="Кумулятивная доходность",
            legend_title_text='Кривая'
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # --- Блок 3: Оптимизация портфеля ---
        if returns.shape[1] > 1:
            st.header("3. Оптимизация портфеля")
            st.divider()
            st.write("На графике 'Граница эффективности' показаны тысячи случайных портфелей. Вы можете найти оптимальный портфель с лучшим соотношением риска и доходности.")
            
            num_portfolios = 5000
            all_weights = np.zeros((num_portfolios, len(tickers)))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            sharpe_arr = np.zeros(num_portfolios)
            
            for x in range(num_portfolios):
                weights_rand = np.array(np.random.random(len(tickers)))
                weights_rand /= np.sum(weights_rand)
                all_weights[x, :] = weights_rand
                port_return = np.sum((returns.mean() * weights_rand) * 252)
                cov_matrix = returns.cov() * 252
                port_volatility = np.sqrt(np.dot(weights_rand.T, np.dot(cov_matrix, weights_rand)))
                ret_arr[x] = port_return
                vol_arr[x] = port_volatility
                sharpe_arr[x] = port_return / port_volatility if port_volatility != 0 else 0

            max_sharpe_idx = sharpe_arr.argmax()
            max_sharpe_port_ret = ret_arr[max_sharpe_idx]
            max_sharpe_port_vol = vol_arr[max_sharpe_idx]
            max_sharpe_weights = all_weights[max_sharpe_idx, :]

            fig_opt = go.Figure(layout=go.Layout(template="plotly_white"))
            fig_opt.add_trace(go.Scatter(
                x=vol_arr, y=ret_arr, mode='markers',
                marker=dict(color=sharpe_arr, colorscale='Reds', showscale=True, colorbar=dict(title="Коэфф. Шарпа")),
                name='Случайные портфели'
            ))
            fig_opt.add_trace(go.Scatter(
                x=[risk], y=[total_return], mode='markers', marker=dict(color=COLOR_TEXT, size=15),
                name='Ваш портфель'
            ))
            fig_opt.add_trace(go.Scatter(
                x=[max_sharpe_port_vol], y=[max_sharpe_port_ret], mode='markers',
                marker=dict(color=COLOR_PRIMARY, size=20, symbol='star'),
                name='Оптимальный портфель'
            ))
            fig_opt.update_layout(
                title="Граница эффективности: поиск оптимального соотношения риска и доходности",
                xaxis_title="Риск (Годовая волатильность)",
                yaxis_title="Доходность (Годовая)",
                legend_title_text='Портфели'
            )
            st.plotly_chart(fig_opt, use_container_width=True)
            
            st.subheader("Рекомендации по весам")
            optimal_weights_df = pd.DataFrame({
                'Акция': tickers,
                'Вес': [f"{w:.2%}" for w in max_sharpe_weights]
            })
            st.dataframe(optimal_weights_df, hide_index=True)

        # --- Блок 4: Прогнозирование и симуляция ---
        st.header("4. Прогноз и симуляция")
        st.divider()
        st.write("С помощью симуляции Монте-Карло мы можем спрогнозировать возможные исходы для вашего портфеля на заданный период. Это поможет оценить потенциальный доход и риски.")
        
        initial_investment = st.number_input("Начальная сумма инвестиций (в ₽):", min_value=1000, value=100000, step=1000, key="initial_investment")
        time_horizon = st.slider("Горизонт прогноза (в днях):", min_value=30, max_value=365, value=90, key="time_horizon")
        num_simulations = st.slider("Количество симуляций:", min_value=100, max_value=5000, value=1000, key="num_simulations")
        
        mean_returns = weighted_returns.mean()
        std_dev = weighted_returns.std()
        
        if std_dev == 0:
            st.warning("Волатильность портфеля равна нулю. Прогноз невозможен.")
            st.stop()

        simulated_end_prices = []
        for x in range(num_simulations):
            future_portfolio_value = initial_investment
            for day in range(time_horizon):
                daily_return = np.random.normal(mean_returns, std_dev)
                future_portfolio_value *= (1 + daily_return)
            simulated_end_prices.append(future_portfolio_value)
        simulated_end_prices = np.array(simulated_end_prices)
        
        st.subheader("Результаты симуляции")
        
        col_pred1, col_pred2 = st.columns(2, gap="large")
        col_pred1.metric("Средний результат", f"{np.mean(simulated_end_prices):,.2f} ₽")
        col_pred2.metric("Медианный результат", f"{np.median(simulated_end_prices):,.2f} ₽")
        
        confidence_level_var = 0.95
        var_value = np.percentile(simulated_end_prices, 100 * (1 - confidence_level_var))
        st.info(f"С вероятностью **95%**, стоимость вашего портфеля не опустится ниже **{var_value:,.2f} ₽** на горизонте {time_horizon} дней.")
        
        fig_sim = go.Figure(layout=go.Layout(template="plotly_white"))
        fig_sim.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Распределение будущей стоимости", marker_color=COLOR_PRIMARY))
        fig_sim.add_vline(x=np.mean(simulated_end_prices), line_dash="dash", line_color=COLOR_SECONDARY, annotation_text="Среднее", annotation_position="top left")
        fig_sim.add_vline(x=var_value, line_dash="dash", line_color=COLOR_TEXT, annotation_text="95% VaR", annotation_position="top right")
        fig_sim.update_layout(
            title=f"Прогноз стоимости портфеля на {time_horizon} дней",
            xaxis_title="Стоимость портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1
        )
        st.plotly_chart(fig_sim, use_container_width=True)
