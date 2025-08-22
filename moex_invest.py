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
PLOTLY_RED = "#D62728"
PLOTLY_GREY = "#808080"
PLOTLY_DARK_GREY = "#4D4D4D"

st.set_page_config(layout="wide")
st.title("Инвестиционный портфель: Анализ и Прогнозирование")
st.markdown("---")

# --- 1. Кэширование данных: Профессиональный подход к API ---

@st.cache_data(ttl=3600)
def get_securities():
    """
    Загружает список ценных бумаг с Moex API.
    
    Returns:
        pd.DataFrame: DataFrame с информацией об акциях или пустой DataFrame в случае ошибки.
    """
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
    """
    Загружает исторические данные по свечам для указанного тикера.
    
    Args:
        ticker (str): Тикер акции.
        start (str): Дата начала в формате 'YYYY-MM-DD'.
        end (str): Дата конца в формате 'YYYY-MM-DD'.
        
    Returns:
        pd.DataFrame: DataFrame с историей цен или пустой DataFrame в случае ошибки.
    """
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

# --- 2. Ввод данных пользователем: Чистый и понятный интерфейс ---

securities_df = get_securities()

if not securities_df.empty:
    tickers = st.multiselect(
        "Выберите акции для портфеля (до 5)",
        securities_df['SECID'],
        default=securities_df['SECID'][:2] if not securities_df.empty else []
    )

    weights = []
    st.write("Задайте доли для каждой акции (в сумме должны быть равны 1.0):")
    cols = st.columns(len(tickers) if tickers else 1)
    
    if tickers:
        for i, ticker in enumerate(tickers):
            with cols[i]:
                weight = st.number_input(
                    f"{ticker}:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/len(tickers) if tickers else 0.0,
                    step=0.01
                )
                weights.append(weight)

    if tickers and (abs(sum(weights) - 1.0) > 0.01):
        st.warning(f"Сумма долей должна быть равна 1.0. Текущая сумма: {sum(weights):.2f}")
        st.stop()

    start_date = st.date_input("Дата начала", datetime.now() - timedelta(days=365))
    end_date = st.date_input("Дата конца", datetime.now())
    st.markdown("---")

    # --- 3. Расчеты и визуализация: Аналитические метрики ---

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
        
        # --- Блок основных метрик ---
        st.header("Основные метрики портфеля")
        st.divider()
        returns = portfolio_df.pct_change().dropna()
        if returns.empty:
            st.warning("Недостаточно данных для расчета доходности.")
            st.stop()

        weighted_returns = returns.dot(weights)
        total_return = (1 + weighted_returns).prod() - 1
        risk = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = total_return / risk if risk != 0 else 0
        
        # Расчет коэффициента Сортино
        downside_returns = weighted_returns[weighted_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = total_return / downside_volatility if downside_volatility != 0 else 0

        # Расчет Беты
        if not imoex_df.empty and 'close' in imoex_df.columns:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            aligned_returns = returns.join(imoex_returns, how='inner', lsuffix='_port', rsuffix='_imoex').dropna()
            
            portfolio_returns_aligned = aligned_returns[tickers].dot(weights)
            market_returns_aligned = aligned_returns['close']
            
            if len(portfolio_returns_aligned) > 1 and len(market_returns_aligned) > 1:
                cov_matrix = np.cov(portfolio_returns_aligned, market_returns_aligned)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            else:
                beta = np.nan
        else:
            beta = np.nan
        
        # Отображение метрик в колонках
        col1, col2, col3, col4, col5 = st.columns(5, gap="large")
        col1.metric("Доходность", f"{total_return:.2%}")
        col2.metric("Риск (Волатильность)", f"{risk:.2%}")
        col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}")
        col4.metric("Коэффициент Сортино", f"{sortino_ratio:.2f}")
        col5.metric("Бета", f"{beta:.2f}" if not np.isnan(beta) else "Недоступно")
        
        st.markdown("---")

        # --- Визуализация динамики цен и доходности ---
        st.header("Динамика доходности и сравнение с индексом")
        st.divider()
        
        fig_price = go.Figure(layout=go.Layout(template="plotly_white"))
        for ticker in tickers:
            fig_price.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df[ticker], name=ticker, line=dict(color=PLOTLY_RED)))
        if not imoex_df.empty:
            fig_price.add_trace(go.Scatter(x=imoex_df.index, y=imoex_df['close'], name='IMOEX', line=dict(dash='dash', color=PLOTLY_GREY)))
        fig_price.update_layout(
            title="Динамика цен активов и индекса",
            xaxis_title="Дата",
            yaxis_title="Цена",
            legend_title_text='Активы'
        )
        st.plotly_chart(fig_price, use_container_width=True)

        cumulative = (1 + weighted_returns).cumprod()
        if not imoex_df.empty:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            imoex_cum = (1 + imoex_returns).cumprod()
            fig_cum = go.Figure(layout=go.Layout(template="plotly_white"))
            fig_cum.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name="Портфель", line=dict(color=PLOTLY_RED)))
            fig_cum.add_trace(go.Scatter(x=imoex_cum.index, y=imoex_cum, name="IMOEX", line=dict(color=PLOTLY_GREY)))
            fig_cum.update_layout(
                title="Кумулятивная доходность портфеля и IMOEX",
                xaxis_title="Дата",
                yaxis_title="Кумулятивная доходность",
                legend_title_text='Кривая'
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # --- Анализ риска и корреляции ---
        st.header("Корреляционный анализ")
        st.divider()
        if returns.shape[1] > 1:
            corr = returns.corr()
            fig_corr = px.imshow(
                corr, text_auto=True, color_continuous_scale=[(0, PLOTLY_GREY), (0.5, "white"), (1, PLOTLY_RED)],
                aspect="auto", title="Корреляционная матрица доходностей", template="plotly_white"
            )
            fig_corr.update_layout(xaxis_title="Активы", yaxis_title="Активы")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Выберите более одной акции для анализа корреляции.")
        
        st.header("Распределение доходности")
        st.divider()
        fig_hist = px.histogram(
            weighted_returns, nbins=50, title="Гистограмма ежедневной доходности портфеля", 
            labels={'value': 'Ежедневная доходность', 'count': 'Частота'}, template="plotly_white"
        )
        fig_hist.update_traces(marker_color=PLOTLY_RED, selector=dict(type='histogram'))
        fig_hist.add_vline(x=0, line_dash="dash", line_color=PLOTLY_GREY)
        st.plotly_chart(fig_hist, use_container_width=True)

        # --- Оптимизация портфеля и граница эффективности ---
        if returns.shape[1] > 1:
            st.header("Оптимизация портфеля: Граница эффективности")
            st.divider()
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
                x=[risk], y=[total_return], mode='markers', marker=dict(color='black', size=15),
                name='Ваш портфель'
            ))
            fig_opt.add_trace(go.Scatter(
                x=[max_sharpe_port_vol], y=[max_sharpe_port_ret], mode='markers',
                marker=dict(color=PLOTLY_RED, size=20, symbol='star'),
                name='Оптимальный портфель'
            ))
            
            fig_opt.update_layout(
                title="Граница эффективности (Efficient Frontier)",
                xaxis_title="Риск (Годовое станд. отклонение)",
                yaxis_title="Доходность (Годовая)",
                legend_title_text='Портфели'
            )
            st.plotly_chart(fig_opt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Оптимальный портфель с максимальным коэффициентом Шарпа")
            
            optimal_weights_df = pd.DataFrame({
                'Акция': tickers,
                'Вес': [f"{w:.2%}" for w in max_sharpe_weights]
            })
            st.dataframe(optimal_weights_df, hide_index=True)

        # --- Новый раздел: Прогнозирование с помощью симуляций Монте-Карло ---
        st.header("Прогноз и оценка риска")
        st.divider()

        initial_investment = st.number_input("Введите начальную сумму инвестиций (в ₽):", min_value=1000, value=100000, step=1000)
        num_simulations = st.slider("Количество симуляций:", min_value=100, max_value=5000, value=1000)
        time_horizon = st.slider("Горизонт прогноза (в днях):", min_value=30, max_value=365, value=90)
        
        # Расчёт основных параметров для симуляции
        mean_returns = weighted_returns.mean()
        std_dev = weighted_returns.std()
        
        if std_dev == 0:
            st.warning("Волатильность портфеля равна нулю. Прогноз невозможен.")
            st.stop()

        # Симуляция методом Монте-Карло
        simulated_end_prices = []

        for x in range(num_simulations):
            future_portfolio_value = initial_investment
            for day in range(time_horizon):
                # Генерируем случайное дневное изменение
                daily_return = np.random.normal(mean_returns, std_dev)
                future_portfolio_value *= (1 + daily_return)
            simulated_end_prices.append(future_portfolio_value)
        
        simulated_end_prices = np.array(simulated_end_prices)
        
        # --- Визуализация и выводы прогноза ---
        st.subheader("Результаты симуляции")
        
        # Расчёт ключевых метрик
        mean_end_price = np.mean(simulated_end_prices)
        median_end_price = np.median(simulated_end_prices)
        
        # Расчёт Value at Risk (VaR)
        confidence_level_var = 0.95
        var_value = np.percentile(simulated_end_prices, 100 * (1 - confidence_level_var))
        
        # Отображение метрик в колонках
        col_pred1, col_pred2 = st.columns(2, gap="large")
        col_pred1.metric("Средний результат", f"{mean_end_price:,.2f} ₽")
        col_pred2.metric("Медианный результат", f"{median_end_price:,.2f} ₽")
        
        st.write(f"**Потенциальный убыток (95% VaR):** За выбранный период, с вероятностью 95%, стоимость вашего портфеля не опустится ниже **{var_value:,.2f} ₽**")
        
        # График распределения результатов
        fig_sim = go.Figure(layout=go.Layout(template="plotly_white"))
        fig_sim.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Распределение будущей стоимости", marker_color=PLOTLY_RED))
        fig_sim.add_vline(x=mean_end_price, line_dash="dash", line_color=PLOTLY_GREY, annotation_text="Среднее", annotation_position="top left")
        fig_sim.add_vline(x=var_value, line_dash="dash", line_color=PLOTLY_DARK_GREY, annotation_text="95% VaR", annotation_position="top right")
        
        fig_sim.update_layout(
            title=f"Распределение прогнозируемой стоимости портфеля за {time_horizon} дней",
            xaxis_title="Стоимость портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # --- АНАЛИЗ УСТОЙЧИВОСТИ К РИСКАМ (СТРЕСС-ТЕСТ) ---
        st.subheader("Анализ устойчивости портфеля (Стресс-тест)")

        st.markdown("Оцените, как ваш портфель поведёт себя в условиях **повышенной рыночной волатильности**.")
        volatility_factor = st.slider("Увеличить волатильность в N раз:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        
        # Симуляция стресс-сценария
        simulated_stress_prices = []
        stress_std_dev = std_dev * volatility_factor

        for x in range(num_simulations):
            future_portfolio_value = initial_investment
            for day in range(time_horizon):
                daily_return = np.random.normal(mean_returns, stress_std_dev)
                future_portfolio_value *= (1 + daily_return)
            simulated_stress_prices.append(future_portfolio_value)
        
        simulated_stress_prices = np.array(simulated_stress_prices)

        # Расчёт метрик для стресс-сценария
        mean_stress_price = np.mean(simulated_stress_prices)
        var_stress = np.percentile(simulated_stress_prices, 100 * (1 - confidence_level_var))

        # Визуализация сравнения
        fig_stress = go.Figure(layout=go.Layout(template="plotly_white"))
        fig_stress.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Нормальный сценарий", marker_color=PLOTLY_RED, opacity=0.7))
        fig_stress.add_trace(go.Histogram(x=simulated_stress_prices, nbinsx=50, name="Стресс-сценарий", marker_color=PLOTLY_GREY, opacity=0.7))
        
        fig_stress.update_layout(
            barmode='overlay',
            title=f"Сравнение распределения стоимости (Нормальный vs Стресс)",
            xaxis_title="Стоимость портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1,
            legend_title_text='Сценарий'
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        # Отображение ключевых выводов
        st.subheader("Выводы по стресс-тесту")
        col_stress1, col_stress2 = st.columns(2)
        col_stress1.metric("Средний результат (Стресс)", f"{mean_stress_price:,.2f} ₽", delta=f"{mean_stress_price - mean_end_price:,.2f} ₽")
        col_stress2.metric("95% VaR (Стресс)", f"{var_stress:,.2f} ₽", delta=f"{var_stress - var_value:,.2f} ₽")
        
        st.info(f"В условиях повышенной волатильности (в {volatility_factor} раз) ожидаемый средний результат снижается, а потенциальный убыток (VaR) увеличивается. Это показывает, насколько ваш портфель устойчив к рыночным потрясениям.")