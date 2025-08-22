import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import apimoex
import requests
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta

main_color = "#D62728"
second_color = "#808080"
back_color = "#FFFFFF"
text_color = "#4D4D4D"

st.set_page_config(layout="wide")
st.title("Анализ и прогнозирование доходности инвестиционного портфеля")
#st.markdown("---")



# Ввод данных пользователем: выбор своего ивестиционного портфеля (акции + их веса в портфеле)
@st.cache_data(ttl=3600)
def get_securities(): #функция, которая загружает список доступных акций с Мосбиржи с основного рынка TQBR
    with requests.Session() as session:
        try:
            data = apimoex.get_board_securities(session, board='TQBR')
            df = pd.DataFrame(data)
            needed_cols = ['SECID', 'SHORTNAME', 'PREVPRICE', 'LOTSIZE', 'FACEVALUE', 'CURRENCYID', 'ISIN']
            existing_cols = [col for col in needed_cols if col in df.columns]
            df = df[existing_cols]
            return df #dataframe с информацией об акциях
        except Exception as e:
            st.error(f"Ошибка при загрузке текущего списка акций: {e}")
            return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_history(ticker, start, end): #функция для получения исторических данных по конкретной акции за указанный период
    with requests.Session() as session:
        try:
            if ticker == "IMOEX":
                data = apimoex.get_market_candles(session, security=ticker, market="index", engine="stock", interval=24, start=start, end=end)
            else:
                data = apimoex.get_market_candles(session, security=ticker, interval=24, start=start, end=end)
            df = pd.DataFrame(data)
            if not df.empty:
                df['begin'] = pd.to_datetime(df['begin'])
                df.set_index('begin', inplace=True) 
            return df
        except Exception as e:
            st.warning(f"Ошибка при загрузке данных для {ticker}: {e}")#формально для тикера
            return pd.DataFrame()
        

securities_df = get_securities() #получаем список тикеров и работаем с ним, если он не пустой
if not securities_df.empty:
    st.sidebar.header("Инвестиционный портфель")
    ticker_to_name = dict(zip(securities_df['SECID'], securities_df['SHORTNAME']))
    
    tickers = st.sidebar.multiselect(
        "Выберите акции для вашего инвестиционного портфеля",
        options=list(ticker_to_name.keys()),
        format_func=lambda x: f"{x} ({ticker_to_name.get(x, 'N/A')})",
        default=securities_df['SECID'][:2].tolist() if not securities_df.empty else []) #пользователь выбирает тикеры 
    
    weights = [] #список для хранения весов акций в портфеле
    if tickers:
        st.sidebar.write("Задайте вес каждой акции в вашем инвестиционном портфеле:")
        if 'weights' not in st.session_state or len(st.session_state.weights) != len(tickers):
            st.session_state.weights = {ticker: 1.0 / len(tickers) for ticker in tickers} #если веса не заданы или их количество не совпадает, устанавливаем равные доли для всех акций

        def update_weights(): #функция, которая нормализует веса акций, чтобы их сумма была равна 1
            total_sum = sum(st.session_state.weights.values())
            if total_sum != 1.0 and total_sum > 0:
                for ticker in st.session_state.weights:
                    st.session_state.weights[ticker] /= total_sum
        

        for ticker in tickers: #создаем ползунок для каждого тикера на левой боковой панели
            st.session_state.weights[ticker] = st.sidebar.slider(
                f"{ticker} ({ticker_to_name.get(ticker, 'N/A')})",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.weights.get(ticker, 0.0),
                step=0.01,
                format="%.2f")
    
        weights = list(st.session_state.weights.values())
        update_weights()
        st.sidebar.info(f"Сумма весов акций в вашем инвестиционном портфеле: **{sum(weights):.2f}**") #выведем подсказку для пользователя в боковой панели, скколько в сумме составляют все веса его акций, чтобы была возможность самостоятельно отрегулировать если что
    #Задаем временной промежуток
    start_date = st.sidebar.date_input("Дата начала инвестиционного анализа", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("Дата конца инвестиционного анализа", datetime.now())
    st.markdown("---")


    # Основная часть
    if tickers: #проводим инвестиционный анализ в случае, если пользователь выбрал хотя бы одну акцию
        portfolio_df = pd.DataFrame() #dataframe, в которй будем добавлять дневные цены закрытия выбранных акций
        for ticker in tickers:
            df = get_history(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not df.empty:
                portfolio_df[ticker] = df['close']
        
        portfolio_df = portfolio_df.dropna(axis=1, how='all')
        valid_tickers = portfolio_df.columns.tolist() #список акций, для которых фактически есть данные после очистки
        if portfolio_df.empty or len(portfolio_df) < 2:
            st.error("Недостаточно данных для выбранных акций или периода. Пожалуйста, измените выбор.")
            st.stop()
        
        imoex_df = get_history('IMOEX', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) #загружаем данные по IMOEX - для сравнения с выбранным пользователем инвестиционным портфелем
        

        # Ключевые показатели портфеля
        st.header("1. Эффективность вашего инвестиционного портфеля")
        st.divider()
        st.write("Этот раздел показывает, насколько успешно ваш инвестиционный портфель работал в прошлом. Здесь вы найдете основные метрики, которые помогут оценить его **доходность и риск**.")
        
        returns = portfolio_df.pct_change().dropna() #рассчитываем дневные доходности как процентное изменение цен, очистка от Nan
        if returns.empty:
            st.warning("Недостаточно данных для расчета доходности.")
            st.stop()
        #Пересобираем веса только для тех акиц, по которым есть данные + нормализуем веса в случае необходимости
        valid_weights = [weights[tickers.index(t)] for t in valid_tickers]
        valid_weights = np.array(valid_weights) / sum(valid_weights)

        weighted_returns = returns.dot(valid_weights) #рассчитываем доходность портфеля как взвешенную сумму доходностей акций
        total_return = (1 + weighted_returns).prod() - 1 #общая доходность портфеля за период (кумулятивный рост)
        risk = weighted_returns.std() * np.sqrt(252) #годовая волатильность
        sharpe_ratio = total_return / risk if risk != 0 else 0 #коэффициент Шарпа (без учёта безрисковой ставки)
        
        downside_returns = weighted_returns[weighted_returns < 0] #только отрицательные доходности
        downside_volatility = downside_returns.std() * np.sqrt(252) #только волатильность вниз 
        sortino_ratio = total_return / downside_volatility if downside_volatility != 0 else 0 #коэффициент Сортино

        #Cинхронизируем доходности инвестиционного портфеля пользователя и индекса IMOEX по выбранным датам
        if not imoex_df.empty and 'close' in imoex_df.columns:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            aligned_returns = returns.join(imoex_returns, how='inner', lsuffix='_port', rsuffix='_imoex').dropna()
            if not aligned_returns.empty and len(aligned_returns) > 1:
                portfolio_returns_aligned = aligned_returns[returns.columns].dot(valid_weights)
                market_returns_aligned = aligned_returns['close']
                cov_matrix = np.cov(portfolio_returns_aligned, market_returns_aligned) #ковариационная матрица между инвестиционным портфелем пользователя и рынком
                if cov_matrix[1, 1] != 0:
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1] #чувствительность портфеля к рынку
        
        #Отображаем полученные метрики + пояснения к метрикам
        col1, col2, col3, col4 = st.columns(4, gap="large")
        col1.metric("Доходность портфеля", f"{total_return:.2%}", help="Общий рост стоимости инвестиционного портфеля за выбранный период.")
        col2.metric("Волатильность доходности портфеля", f"{risk:.2%}", help="Изменчивость стоимости инвестиционного портфеля.")
        col3.metric("Коэффициент Шарпа", f"{sharpe_ratio:.2f}", help="Показатель эффективности, который показывает, какую доходность приносят инвестиции на каждую единицу риска.")
        col4.metric("Коэффициент Сортино", f"{sortino_ratio:.2f}", help="Показатель эффективности, который измеряет доходность инвестиций с поправкой на риск, учитывая только отрицательные отклонения доходности.")

        
        # Динамика и сравнение инвестиционного портфеля пользователя с рынком
        st.header("2. Динамика вашего инвестиционного портфеля")
        st.divider()
        st.write("Этот график наглядно показывает, как стоимость вашего инвестиционного портфеля изменялась в течение заданного вами временного промежутка. Вы также можете сравнить его с **Индексом МосБиржи (IMOEX)**, чтобы понять, опережает ли ваша стратегия рынок.")
        
        cumulative = (1 + weighted_returns).cumprod() #кумулятивная доходность портфеля
        fig_cum = go.Figure(layout=go.Layout(template="plotly_dark"))
        fig_cum.add_trace(go.Scatter(x=cumulative.index, y=cumulative, name="Ваш инвестиционный портфель", line=dict(color=main_color, width=2)))
        
        if not imoex_df.empty:
            imoex_returns = imoex_df['close'].pct_change().dropna()
            imoex_cum = (1 + imoex_returns).cumprod()
            fig_cum.add_trace(go.Scatter(x=imoex_cum.index, y=imoex_cum, name="IMOEX", line=dict(color=second_color, dash='dash', width=2)))
        
        fig_cum.update_layout(
            title="Сравнение кумулятивной доходности вашего инвестиционного с индексом Мосбиржи IMOEX",
            xaxis_title="Дата",
            yaxis_title="Рост портфеля (в долях)",
            legend_title_text='Кривая')
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Попытка оптимизации инвестиционного портфеля
        if returns.shape[1] > 1:
            st.header("3. Оптимизация вашего инвестиционного портфеля")
            st.divider()
            st.write("На этом графике вы видите **границу эффективности** — множество инвестиционных портфелей с идеальным соотношением риска и доходности (иначе говоря набор инвестиционных портфелей, которые обеспечивают максимальную ожидаемую доходность при заданном уровне риска или минимальный риск при заданной ожидаемой доходности). Ваша задача заключается в том, чтобы найти на этом графике инвестиционный портфель, который лучше всего соответствует вашим целям.")
            st.markdown("""
            **Случайные инвестиционные портфели** образуют облако. Каждый портфель в нём — это одна из тысяч случайных комбинаций акций.  **Ваш портфель** показан чёрной точкой.
            
            **Оптимальный инвестиционный портфель** показан красной звездой. Это самая эффективная комбинация активов, которая даёт максимальную доходность при вашем уровне риска.
            """)
            
            #Подготовка массивов для 5000 случайных инвестиционных портфелей
            num_portfolios = 5000
            all_weights_opt = np.zeros((num_portfolios, len(returns.columns)))
            ret_arr = np.zeros(num_portfolios)
            vol_arr = np.zeros(num_portfolios)
            sharpe_arr = np.zeros(num_portfolios)
            
            for x in range(num_portfolios):
                weights_rand = np.array(np.random.random(len(returns.columns)))
                weights_rand /= np.sum(weights_rand)
                all_weights_opt[x, :] = weights_rand #генерация случайных весов, сумма которых 1
                port_return = np.sum((returns.mean() * weights_rand) * 252) #годовая доходность портфеля
                cov_matrix = returns.cov() * 252 
                port_volatility = np.sqrt(np.dot(weights_rand.T, np.dot(cov_matrix, weights_rand))) #годовая волатильность портфеля
                ret_arr[x] = port_return
                vol_arr[x] = port_volatility
                sharpe_arr[x] = port_return / port_volatility if port_volatility != 0 else 0
            #Ищем портфель с максимальным коэффициентом Шарпа
            max_sharpe_idx = sharpe_arr.argmax()
            max_sharpe_port_ret = ret_arr[max_sharpe_idx]
            max_sharpe_port_vol = vol_arr[max_sharpe_idx]
            max_sharpe_weights = all_weights_opt[max_sharpe_idx, :]

            fig_opt = go.Figure(layout=go.Layout(template="plotly_dark"))
            fig_opt.add_trace(go.Scatter(
                x=vol_arr, y=ret_arr, mode='markers',
                marker=dict(color=sharpe_arr, colorscale='Reds', showscale=True, colorbar=dict(title="Коэффициент Шарпа")),
                name='Случайные портфели'))
            
            fig_opt.add_trace(go.Scatter(
                x=[risk], y=[total_return], mode='markers', marker=dict(color=text_color, size=15),
                name='Ваш портфель'))
            
            fig_opt.add_trace(go.Scatter(
                x=[max_sharpe_port_vol], y=[max_sharpe_port_ret], mode='markers',
                marker=dict(color=main_color, size=20, symbol='star'),
                name='Оптимальный портфель'))
            
            fig_opt.update_layout(
                title="Оптимизация вашего инвестиционного портфеля: поиск идеального соотношения риска и доходности",
                xaxis_title="Годовая волатильность портфеля)",
                yaxis_title="Годовая доходность портфеля",
                legend_title_text='Инвестиционные портфели', legend=dict(x=1.1, y=0.5))
            
            st.plotly_chart(fig_opt, use_container_width=True)
            st.subheader("Рекомендованные веса акций для оптимального инвестиционного портфеля")
            optimal_weights_df = pd.DataFrame({
                'Акция': returns.columns,
                'Вес акции в портфеле': [f"{w:.2%}" for w in max_sharpe_weights]})
            st.dataframe(optimal_weights_df, hide_index=True)
            
            
            # Кореляционый аналтз
            st.header("4. Анализ корреляции активов в вашем инвестиционном портфеле")
            st.divider()
            st.write("Корреляция показывает, как доходности ваших активов движутся относительно друг друга. **Низкая корреляция** — это признак оптимальности стратегии, так как это означает, что если один актив падает в цене, то цена другого может расти, тем самым **уменьшая общий риск** инвестиционного портфеля.")
            
            #Тепловая карта корреляций
            corr = returns.corr()
            fig_corr = px.imshow(
                corr, text_auto=True, color_continuous_scale=[(0, second_color), (0.5, "white"), (1, main_color)],
                aspect="auto", title="Матрица корреляции доходностей активов в вашем инвестиционном портфеле", template="plotly_dark")
            
            fig_corr.update_layout(xaxis_title="Активы в портфеле", yaxis_title="Активы в портфеле")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Выводы по корреляции")
            if corr.min().min() > 0.5:
                st.warning("**Сильная корреляция.** Все активы в вашем портфеле движутся почти одинаково. Это делает его уязвимым к падениям всего рынка и повышает риски убытка.")
            elif corr.max().max() < 0.2:
                st.success("**Низкая корреляция.** Отличная диверсификация: падение цены одного актива будет меньше влиять на доходность всего портфеля.")
            else:
                st.info("**Умеренная корреляция.** Портфель обладает хорошей диверсификацией. Это обеспечивает баланс между риском и доходностью.")

        # Прогнозирование доходности и устойчивости инвестиционного портфеля
        st.header("5. Прогноз доходности вашего инвестиционного портфеля методом Монте-Карло)")
        st.divider()
        st.write("Метод **Монте-Карло** — это математический инструмент, который симулирует тысячи возможных будущих сценариев. Вместо того чтобы пытаться угадать одну точную цену, он показывает **спектр возможных исходов** и помогает оценить риски.")
        st.write("Как же работает этот метод?:")
        st.markdown("""
        1. **Исторические данные:** Модель анализирует историческую доходность и волатильность вашего инвестиционного портфеля.
        2. **Случайные сценарии:** Модель генерирует тысячи случайных исходов для вашего инвестиционного портфеля, используя исторические данные как основу.
        3. **Распределение результатов:** В итоге вы видите, где с большей вероятностью окажется ваш инвестиционный портфель и какие риски вас поджидают.
        """)
        
        #Пользователь задаёт параметры симуляции
        initial_investment = st.number_input("Начальная сумма инвестиций (в ₽):", min_value=1000, value=100000, step=1000, key="initial_investment")
        time_horizon = st.slider("Горизонт прогноза (в днях):", min_value=30, max_value=365, value=90, key="time_horizon")
        num_simulations = st.slider("Количество симуляций:", min_value=100, max_value=5000, value=1000, key="num_simulations")
        
        mean_returns = weighted_returns.mean()
        std_dev = weighted_returns.std()
        
        if std_dev == 0: #проверка на нулевую волатильность
            st.warning("Волатильность вашего инвестиционного портфеля равна нулю. Прогноз невозможен.")
            st.stop()

        simulated_end_prices = []
        #Сам метод: множество сценариев для инвестиционного портфеля с нормальным распределением доходности
        for x in range(num_simulations):
            future_portfolio_value = initial_investment
            for day in range(time_horizon):
                daily_return = np.random.normal(mean_returns, std_dev)
                future_portfolio_value *= (1 + daily_return)
            simulated_end_prices.append(future_portfolio_value)
        simulated_end_prices = np.array(simulated_end_prices)
        mean_end_price = np.mean(simulated_end_prices)

        st.subheader("Результаты прогнозирования доходности вашего инвестиционного портфеля")
        col_pred1, col_pred2 = st.columns(2, gap="large")
        col_pred1.metric("Средний результат", f"{mean_end_price:,.2f} ₽")
        col_pred2.metric("Медианный результат", f"{np.median(simulated_end_prices):,.2f} ₽")
        
        confidence_level_var = 0.95
        var_value = np.percentile(simulated_end_prices, 100 * (1 - confidence_level_var)) #расчёт VaR - нижняя граница потерь с 95% доверительной вероятностью
        st.info(f"С вероятностью **95%**, стоимость вашего инвестиционного портфеля не опустится ниже **{var_value:,.2f} ₽** на горизонте {time_horizon} дней.")
        
        #Гистограмма распределения исходов для инвестиционного портфеля
        fig_sim = go.Figure(layout=go.Layout(template="plotly_dark"))
        fig_sim.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Распределение будущей стоимости вашего инвестиционного портфеля", marker_color=main_color))
        fig_sim.add_vline(x=mean_end_price, line_dash="dash", line_color=second_color, annotation_text="Среднее", annotation_position="top left")
        fig_sim.add_vline(x=var_value, line_dash="dash", line_color=text_color, annotation_text="95% VaR", annotation_position="top right")
        fig_sim.update_layout(
            title=f"Прогноз стоимости вашего инвестиционного портфеля на {time_horizon} дней",
            xaxis_title="Стоимость инвестиционного портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1)
        st.plotly_chart(fig_sim, use_container_width=True)

        st.header("6. Анализ устойчивости вашего инвестиционного портфеля")
        st.divider()
        st.write("Стресс-тест позволяет оценить, как ваш инвестиционный портфель поведет себя в условиях **повышенной рыночной волатильности**, например, во время кризиса. Мы удваиваем историческую волатильность, чтобы смоделировать такой сценарий и понять, насколько ваш инвестиционный портфель устойчив к резким падениям рынка.")
        
        volatility_factor = 2.0 #иммитация кризиса - х2 волатильность
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

        fig_stress = go.Figure(layout=go.Layout(template="plotly_dark"))
        fig_stress.add_trace(go.Histogram(x=simulated_end_prices, nbinsx=50, name="Нормальный сценарий", marker_color=main_color, opacity=0.7))
        fig_stress.add_trace(go.Histogram(x=simulated_stress_prices, nbinsx=50, name="Стресс-сценарий (удвоенная волатильность)", marker_color=second_color, opacity=0.7))
        fig_stress.update_layout(
            barmode='overlay',
            title=f"Сравнение распределения стоимости вашего инвестиционного портфеля при нормальном сценарии и стресс-сценарии)",
            xaxis_title="Стоимость инвестиционного портфеля (₽)",
            yaxis_title="Количество симуляций",
            bargap=0.1,
            legend_title_text='Сценарий')
        st.plotly_chart(fig_stress, use_container_width=True)

        st.subheader("Результаты стресс-тестирования")
        col_stress1, col_stress2 = st.columns(2)
        col_stress1.metric("Средний результат (Стресс-сценарий)", f"{mean_stress_price:,.2f} ₽", delta=f"{mean_stress_price - mean_end_price:,.2f} ₽")
        col_stress2.metric("95% Value at Risk (Стресс-сценарий)", f"{var_stress:,.2f} ₽", delta=f"{var_stress - var_value:,.2f} ₽")
        
        st.info("Результаты стресс-теста показывают, насколько ваш инвецстиционный портфель устойчив к рыночным потрясениям. В условиях повышенной волатильности потенциальные убытки значительно возрастают.")
