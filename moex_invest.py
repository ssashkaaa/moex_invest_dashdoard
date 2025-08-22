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
    ticker_to_name = dict(zip(securities_df['SECID'], securities_df['SHORTNAME']))
    
    tickers = st.sidebar.multiselect(
        "Выберите акции для портфеля (до 5)",
        options=list(ticker_to_name.keys()),
        format_func=lambda x: f"{x} ({ticker_to_name.get(x, 'N/A')})",
        default=securities_df['SECID'][:2].tolist() if not securities_df.empty else []
    )
    
    weights = []
    if tickers:
        st.sidebar.write("Задайте доли для каждой акции:")
        # Инициализация весов в session_state, если их нет
        if 'weights' not in st.session_state or len(st.session_state.weights) != len(tickers):
            st.session_state.weights = {ticker: 1.0 / len(tickers) for ticker in tickers}

        # Обновление весов
        def update_weights():
            total_sum = sum(st.session_state.weights.values())
            if total_sum != 1.0 and total_sum > 0:
                for ticker in st.session_state.weights:
                    st.session_state.weights[ticker] /= total_sum

        for ticker in tickers:
            st.session_state.weights[ticker] = st.sidebar.slider(
                f"{ticker} ({ticker_to_name.get(ticker, 'N/A')})",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights.get(ticker, 0.0),
                step=0.01,
                format="%.2f"
            )
        weights = list(st.session_state.weights.values())
        update_weights()
        st.sidebar.info(f"Сумма долей: **{sum(weights):.2f}**")
    
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
        
        portfolio_df = portfolio_df.dropna(axis=1, how='all')
        valid_tickers = portfolio_df.columns.tolist()
        
        if portfolio_df.empty or len(portfolio_df) < 2:
            st.error("Недостаточно данных для выбранных акций или периода. Пожалуйста, измените выбор.")
            st.stop()
        
        imoex_df = get_history('IMOEX', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # --- Блок 1: Ключевые показатели портфеля ---
        st.header("1. Эффективность портфеля")
        st.divider()
        st.write("Этот раздел представляет **ключевые метрики** для оценки вашего портфеля. **Доходность** показывает, насколько выросла его стоимость, а **Риск (Волатильность)** — как сильно она колеблется.")
        st.write("— **Коэффициент Шарпа** измеряет доходность на единицу риска: чем выше значение, тем лучше. \n— **Коэффициент Сортино** похож на Шарпа, но учитывает только риск падения, игнорируя положительные колебания. \n— **Бета** показывает, как ваш портфель движется по отношению к рынку (индексу МосБиржи): Бета > 1 означает, что портфель более волатилен, чем рынок.")

        returns = portfolio_df.pct_change().dropna()
        if returns.empty:
            st.warning("Недостаточно данных для расчета доходности.")
            st.stop()

        valid_weights = [weights[tickers.index(t)] for t in valid_tickers]
        valid_weights = np.array(valid_weights) / sum(valid_weights)

        weighted_returns = returns.dot(valid_weights)
        total_return = (1 + weighted_returns).prod() - 1
        risk = weighted_returns.std() * np.sqrt(252)
        sharpe_ratio = total_return / risk if risk != 0 else 0
        
        downside_returns = weighted_returns[weighted_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = total_return / downside_volatility if downside_volatility != 0 else 0

        beta = np.nan
        if not imoex_df.empty and 'close' in imoex_df.columns:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            aligned_returns = returns.join(imoex_returns, how='inner', lsuffix='_port', rsuffix='_imoex').dropna()
            if not aligned_returns.empty and len(aligned_returns) > 1:
                portfolio_returns_aligned = aligned_returns[returns.columns].dot(valid_weights)
                market_returns_aligned = aligned_returns['close']
                cov_matrix = np.cov(portfolio_returns_aligned, market_returns_aligned)
                if cov_matrix[1, 1] != 0:
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        col1, col2, col3, col4, col5 = st.columns(5, gap="large")
        col1.metric("Доходность", f"{total_return:.2%}")
        col2.metric("Риск (Волатильность)", f"{risk:.2%}")
        col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}")
        col4.metric("Коэффициент Сортино", f"{sortino_ratio:.2f}")
        col5.metric("Бета", f"{beta:.2f}" if not np.isnan(beta) else "Недоступно")
        
        # --- Блок 2: Динамика и сравнение с рынком ---
        st.header("2. Динамика и сравнение с индексом")
        st.divider()
        st.write("Этот график показывает, как стоимость вашего портфеля изменялась во времени по сравнению с рыночным индексом. Это ключевой показатель, который помогает понять, **опережает ли ваша стратегия рынок** или отстаёт от него.")
        
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
            st.write("На графике **'Граница эффективности'** показаны тысячи случайно сгенерированных портфелей. Цель оптимизации — найти портфель с лучшим соотношением доходности и риска.")
            st.write("— **Случайные портфели** образуют 'облако'. \n— **Ваш портфель** — черная точка на этом графике. \n— **Оптимальный портфель** — это красная звезда (*). Это идеальная комбинация активов, которая даёт максимальную доходность при текущем уровне риска, или минимальный риск при текущей доходности.")
            
            num_portfolios = 5000
            all_weights_opt = np.zeros((num_portfolios, len(returns.columns)))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            sharpe_arr = np.zeros(num_portfolios)
            
            for x in range(num_portfolios):
                weights_rand = np.array(np.random.random(len(returns.columns)))
                weights_rand /= np.sum(weights_rand)
                all_weights_opt[x, :] = weights_rand
                port_return = np.sum((returns.mean() * weights_rand) * 252)
                cov_matrix = returns.cov() * 252
                port_volatility = np.sqrt(np.dot(weights_rand.T, np.dot(cov_matrix, weights_rand)))
                ret_arr[x] = port_return
                vol_arr[x] = port_volatility
                sharpe_arr[x] = port_return / port_volatility if port_volatility != 0 else 0

            max_sharpe_idx = sharpe_arr.argmax()
            max_sharpe_port_ret = ret_arr[max_sharpe_idx]
            max_sharpe_port_vol = vol_arr[max_sharpe_idx]
            max_sharpe_weights = all_weights_opt[max_sharpe_idx, :]

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
                'Акция': returns.columns,
                'Вес': [f"{w:.2%}" for w in max_sharpe_weights]
            })
            st.dataframe(optimal_weights_df, hide_index=True)
            
            # --- Блок 4: Анализ корреляции ---
            st.header("4. Анализ корреляции активов")
            st.divider()
            st.write("Корреляция показывает, как доходности ваших активов движутся относительно друг друга. **Низкая корреляция** — это признак хорошей диверсификации, так как падение одного актива может компенсироваться ростом другого.")
            
            corr = returns.corr()
            fig_corr = px.imshow(
                corr, text_auto=True, color_continuous_scale=[(0, COLOR_SECONDARY), (0.5, "white"), (1, COLOR_PRIMARY)],
                aspect="auto", title="Матрица корреляции доходностей", template="plotly_white"
            )
            fig_corr.update_layout(xaxis_title="Активы", yaxis_title="Активы")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Выводы по корреляции")
            if corr.min().min() > 0.5:
                st.warning("**Сильная корреляция.** Активы вашего портфеля движутся в одном направлении. Это означает, что портфель может быть менее устойчив к падениям, так как все активы могут падать одновременно.")
            elif corr.max().max() < 0.2:
                st.success("**Низкая корреляция.** Ваши активы слабо коррелируют или имеют отрицательную корреляцию, что является признаком хорошей диверсификации. Это снижает общий риск портфеля, так как падение одного актива может компенсироваться ростом другого.")
            else:
                st.info("**Умеренная корреляция.** Портфель обладает умеренной диверсификацией. Это обеспечивает баланс между риском и доходностью.")

        # --- Блок 5: Прогнозирование и симуляция ---
        st.header("5. Прогноз и симуляция (Метод Монте-Карло)")
        st.divider()
        st.write("Метод **Монте-Карло** — это математическая модель, которая многократно симулирует тысячи возможных будущих сценариев на основе исторических данных. Это позволяет оценить не один конкретный результат, а **диапазон возможных исходов** и рассчитать потенциальные риски.")
        st.write("Как это работает:")
        st.markdown("""
        1. **Исторические данные:** Модель анализирует вашу историческую доходность и волатильность.
        2. **Случайные симуляции:** Затем она запускает тысячи симуляций. В каждой симуляции генерируется случайный ежедневный доход, который соответствует исторической волатильности.
        3. **Диапазон результатов:** В итоге вы получаете не одну цифру, а распределение возможных будущих цен. Это помогает понять, какой результат наиболее вероятен, а какой — наихудший.
        """)
        
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

        st.header("6. Анализ устойчивости (Стресс-тест)")
        st.divider()
        st.write("Стресс-тест позволяет оценить, как ваш портфель поведет себя в условиях **повышенной рыночной волатильности**, например, во время кризиса. Мы удваиваем историческую волатильность, чтобы смоделировать такой сценарий и понять, насколько ваш портфель устойчив к резким падениям.")
        
        volatility_factor = 2.0
        simulated_stress_prices = []
        stress_std_dev = std_dev * volatility_factor

        for x in range(num_simulations):
            future_portfolio_value = initial_investment
            for day in range(time_horizon):
                daily_return = np.random.normal(mean_returns, stress_std_dev)
                future_portfolio_value *= (1 + daily_return)
            simulated_stress_prices.append(future_portfolio_value)
        simulated_stress_prices = np.array(simulated_stress_prices)

        mean_stress_price = np.mean(simulated_stress_prices)
        var_stress = np.percentile(simulated_stress_prices, 100 * (1 - confidence_level_var))

        fig_stress = go.Figure(layout=go.Layout(template="plotly_white"))
        fig_stress.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Нормальный сценарий", marker_color=COLOR_PRIMARY, opacity=0.7))
        fig_stress.add_trace(go.Histogram(x=simulated_stress_prices, nbinsx=50, name="Стресс-сценарий (волатильность x2)", marker_color=COLOR_SECONDARY, opacity=0.7))
        fig_stress.update_layout(
            barmode='overlay',
            title=f"Сравнение распределения стоимости (Нормальный vs Стресс-сценарий)",
            xaxis_title="Стоимость портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1,
            legend_title_text='Сценарий'
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        st.subheader("Выводы по стресс-тесту")
        col_stress1, col_stress2 = st.columns(2)
        col_stress1.metric("Средний результат (Стресс)", f"{mean_stress_price:,.2f} ₽", delta=f"{mean_stress_price - mean_end_price:,.2f} ₽")
        col_stress2.metric("95% VaR (Стресс)", f"{var_stress:,.2f} ₽", delta=f"{var_stress - var_value:,.2f} ₽")
        
        st.info("Результаты стресс-теста показывают, насколько ваш портфель устойчив к рыночным потрясениям. В условиях повышенной волатильности потенциальные убытки значительно возрастают.")
