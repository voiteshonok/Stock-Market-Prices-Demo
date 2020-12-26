import os
import datetime
import numpy as np
import pandas as pd
import time
import datetime
import joblib
import xgboost
from matplotlib import pyplot as plt
import plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import altair as alt


DATA_PATH = 'data/'
MODELS_PATH = 'models/'

COMPANY_NAMES_TO_STOCK_NAMES = {'Cern': 'cern', 'IBM': 'ibm', 'Yandex': 'yndx'}


def get_data_frame_from_tigger(ETF_NAME):
    ETF_DIRECTORY = "data"
    df = pd.read_csv(os.path.join(ETF_DIRECTORY, ETF_NAME.lower() + '.us.txt'), sep=',')
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= datetime.datetime(2010, 1, 1))]
    df = df[(df["Date"] <= datetime.datetime(2017, 12, 31))]
    df.set_index(pd.Series(range(0, len(df))), inplace=True)
    return df


def RSI(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    prices_up = delta.copy()
    prices_down = delta.copy()
    prices_up[prices_up < 0] = 0
    prices_down[prices_down > 0] = 0
    roll_up = prices_up.rolling(n).mean()
    roll_down = prices_down.abs().rolling(n).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def stochastic(df, k, d):
    df = df.copy()
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=d).mean()
    return df


def get_processed_test_data(df):
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()
    df['RSI'] = RSI(df).fillna(0)

    ema_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    ema_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(ema_12 - ema_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
    df['Close'] = df['Close'].shift(-1)
    df = df.iloc[33:]
    df = df[:-1]
    df.index = range(len(df))

    test_df = df[(df['Date'] >= datetime.datetime(2016, 11, 1))].copy()
    test_df = test_df[(test_df['Date'] <= datetime.datetime(2017, 10, 31))]

    drop_cols = ['Volume', 'Open', 'Low', 'High', 'OpenInt']
    test_df = test_df.drop(drop_cols, 1)

    return test_df

def load_company_model(stock_name):
    model = joblib.load(MODELS_PATH + stock_name + '_model.pkl')
    return model


def load_company_data(stock_name):
    df = pd.read_csv(DATA_PATH + stock_name + '.us.txt', parse_dates=['Date'])
    return df

def load_data_for_predicted_prices_plot(stock_names: list):
    data = pd.DataFrame(columns=['symbol', 'date', 'predicted_price', 'actual_price'])
    for stock_name in stock_names:
        model = load_company_model(stock_name)
        test_data = load_company_data(stock_name)
        processed_test_data = get_processed_test_data(test_data)
        date = processed_test_data['Date'].dt.date
        date = date.reset_index(drop=True)
        x_test_data = processed_test_data.drop(['Date', 'Close'], axis=1)
        predicted_data = pd.DataFrame(model.predict(x_test_data))
        predicted_data = predicted_data.rename(columns={0: 'predicted_price'})
        symbol_column = [stock_name.upper()] * predicted_data.shape[0]
        predicted_data.insert(0, 'date', date)
        predicted_data.insert(0, 'symbol', symbol_column)
        data = pd.concat([data, predicted_data])
    return data


def load_data_for_predicted_actual_prices_plot(stock_name: str):
    model = load_company_model(stock_name)
    test_data = load_company_data(stock_name)
    processed_test_data = get_processed_test_data(test_data)
    date = processed_test_data['Date'].dt.date
    date = date.reset_index(drop=True)
    x_test_data = processed_test_data.drop(['Date', 'Close'], axis=1)
    predicted_data = pd.DataFrame(model.predict(x_test_data))
    predicted_data.rename(columns={0: 'price'}, inplace=True)
    predicted_data['price_type'] = 'predicted_price'
    predicted_data.insert(0, 'date', date)

    actual_data = pd.DataFrame(processed_test_data['Close'])
    actual_data['price_type'] = 'actual_price'
    actual_data = actual_data.reset_index(drop=True)
    actual_data.rename(columns={'Close': 'price'}, inplace=True)
    actual_data.insert(0, 'date', date)
    data = pd.concat([predicted_data, actual_data])

    return data


def create_list_of_stock_names(company_names: list):
    stock_names = []
    for company_name in company_names:
        stock_names.append(COMPANY_NAMES_TO_STOCK_NAMES[company_name])
    return stock_names


def main():
    st.title("Stock Market Prices Demo")
    nav = st.sidebar.radio("Navigation", ["Introduction", "Feature Engineering", "Prediction"])

    if nav == "Introduction":
        ETF_NAME = 'CERN'
        df = get_data_frame_from_tigger(ETF_NAME)

        st.header("Introduction")

        st.subheader("Business task")
        st.markdown("")

        st.subheader("Dataset")
        st.markdown('''We will use the [Huge Stock Market Dataset]
            (https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs). 
            High-quality financial data is expensive to acquire. Therefore, such data is rarely 
            shared for free. The full historical daily prices and volume data for all US-based 
            stocks and ETFs trading on the NYSE, NASDAQ, and NYSE MKT are provided. The dataset 
            includes a lot of different companies. So, to show how our model works, we chose 
            only some of them: Ford, Yandex, IBM, etc.''')

        st.subheader("Content")
        st.markdown('''The data is presented in CSV format as follows: Date, Open, High, Low, 
            Close, Volume, OpenInt. We will train the model on data from 2010 to 2016 because 
            other data is way too old and has no significant information for the 2010s decade. 
            The prediction will be built in 2017. Note that prices have been adjusted for dividends 
            and splits. To demonstrate how data looks like, we will select CERN. There you can see 
            the head of the dataset:''')

        st.dataframe(df.head())
        st.write("Let's analyze the description. This is the structure. It has ‘Date’ as the index feature.\n"
                 "‘High’ denotes the highest value of the day. ‘Low’ denotes the lowest. ‘Open’ is the opening\n"
                 "Price and ‘Close’ is the closing for that Date. Now, sometimes close values are regulated\n"
                 "by the companies. So the final value is the ‘Adj Close’ which is the same as ‘Close’ Value\n"
                 "if the stock price is not regulated. ‘Volume’ is the amount of Stock of that company traded\n"
                 "on that date.")

        st.subheader("Plotting dataset")
        st.markdown("On the chart below you can see how CERN stock prices changed from 2010 to 2016.")
        df_intro = df[["Date", "Open", "High", "Low", "Close"]]
        df_intro = df_intro[(df["Date"] <= datetime.datetime(2011, 12, 31))]
        df_intro.set_index("Date", inplace=True)
        st.line_chart(df_intro)

    if nav == "Feature Engineering":
        width = 1000
        height = 500

        st.header("Feature Engineering")

        st.subheader("Historical ETF prices")
        st.markdown('''Data frame with historial prices for fund consists of 7 columns
             which are: *Date*, *Open/High/Low/Close* prices, *Volume* count and *Open Interest* number. *OpenInt column* has
              only 0 values, so I will just ignore it and focus on the rest of information. In tables below you can
               see sample prices from the data frame and also few statistics about each column e.g. min/max values,
                standard deviation etc.''')

        option = st.selectbox("What company ? ", ["CERN", "IBM", "YNDX"])

        df = get_data_frame_from_tigger(option)

        if st.checkbox("Show Head"):
            st.dataframe(df.head())

        if st.checkbox("Show Description"):
            st.dataframe(df.describe())

        st.subheader("I. OHLC Chart")
        st.markdown('''An OHLC chart shows the *open, high, low and close* prices of a stock. It shows you how
             the price was changing during a particular day and give you a sense of e.g. momentum or volatility of stock.
              The tip of the lines represent the low and high values and the horizontal segments represent the open and
               close values. Sample points where the close value is higher (lower) then the open value are called
                increasing (decreasing). By default, increasing items are drawn in green whereas decreasing are drawn
                 in red.''')

        fig = go.Figure([go.Ohlc(x=df.Date,
                                 open=df.Open,
                                 high=df.High,
                                 low=df.Low,
                                 close=df.Close)])
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("II. Volume")
        st.markdown('''A *volume* is a very basic measure that shows a number of shares traded (bought, sold) over
             a certain period of time e.g. daily. It is such a simple but often overlooked indicator.
              *Volume* is so important because it basically represents the activity in stock trading.
               Higher volume value indicates higher interests in trading a stock.''')
        st.markdown('*2012-2013*')

        # fig = go.Figure(go.Bar(x=df.Date, y=df.Volume, name='Volume', marker_color='red'))
        df['Date'] = pd.to_datetime(df['Date'])
        fig = go.Figure(
            go.Bar(x=df[(df['Date'].dt.year >= 2012) & (df['Date'].dt.year <= 2013)].Date,
                   y=df.Volume, name='Volume',
                   marker_color='red'))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("III. Moving Averages")
        st.markdown('''Moving Averages (MA) help to smooth out stock prices on a chart by filtering out
                    short-term price fluctuations. We calculate moving averages over a defined period
                    of time e.g. last 9, 50 or 200 days. There are two (most common) averages used in
                    technical analysis which are:''')
        st.markdown('''\t•Simple Moving Average (SMA) - a simple average calculated over last N days e.g. 50, 100 or 200
                    \t•Exponential Moving Average (EMA) - an average where greater weights are applied to recent prices''')

        df['EMA_9'] = df['Close'].ewm(5).mean().shift()
        df['SMA_50'] = df['Close'].rolling(50).mean().shift()
        df['SMA_100'] = df['Close'].rolling(100).mean().shift()
        df['SMA_200'] = df['Close'].rolling(200).mean().shift()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_50, name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_100, name='SMA 100'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_200, name='SMA 200'))
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', line_color='dimgray', opacity=0.3))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("IV. RSI")
        st.markdown('''Another commonly used indicator is a Relative Strength Index (RSI) that indicates magnitude
             of recent price changes. It can show that a stock is either overbought or oversold.
              Typically RSI value of 70 and above signal that a stock is becoming overbought/overvalued, meanwhile value
               of 30 and less can mean that it is oversold. Full range of RSI is from 0 to 100.''')

        num_days = 365
        df['RSI'] = RSI(df).fillna(0)
        fig = go.Figure(go.Scatter(x=df.Date.tail(num_days), y=df.RSI.tail(num_days)))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("V. MACDI")
        st.markdown('''Moving Average Convergence Divergence (MACD) is an indicator which shows the relationship
             between two exponential moving averages i.e. 12-day and 26-day EMAs. We obtain MACD by substracting 26-day
              EMA (also called slow EMA) from the 12-day EMA (or fast EMA).''')

        EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
        EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
        MACD = pd.Series(EMA_12 - EMA_26)
        MACD_signal = pd.Series(MACD.ewm(span=9, min_periods=9).mean())

        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=MACD, name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.Date, y=MACD_signal, name='Signal line'), row=2, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("VI. Stochastic")
        st.markdown('''The last technical tool in this notebook is a stochastic oscillator
                                is quite similar to RSI in the sense that it's values (also in range 0-100)
                                can indicate whether a stock is overbought/oversold or not. It is arguably
                                the most complicated indicator compared to the ones introduced earlier.
                                Stochastic can be calculated as:''')
        st.latex(r'''\%K = (\frac{C - L_{14}}{H_{14} - L_{14}}) \times 100''')
        st.markdown('''where: **C** is the most recent close price, **L** and **H** are the
                        lowest/highest prices traded in last 14 days.''')
        st.markdown('''This  **%𝐾**  stochastic is often referred as the *"slow stochastic indicator".
                        There is also a *"fast stochastic indicator" that can be obtained as:''')
        st.latex(r'''\%D = SMA_{3}(\%K)''')

        stochs = stochastic(df, k=14, d=3)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_k.tail(365), name='K stochastic'))
        fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_d.tail(365), name='D stochastic'))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

    if nav == "Prediction":
        st.title('Predict stock prices')
        st.write("Here's our first attempt at using data to create a table:")

        st.write('Showing predicted prices for some companies')

        company_names = st.multiselect('Choose company name(s):', sorted(COMPANY_NAMES_TO_STOCK_NAMES.keys()),
                                       default=[sorted(COMPANY_NAMES_TO_STOCK_NAMES.keys())[0]])
        stock_names = create_list_of_stock_names(company_names)
        data_predicted_prices = load_data_for_predicted_prices_plot(stock_names)

        highlight_predicted_prices = alt.selection(type='single', on='mouseover',
                                                   fields=['symbol'], nearest=True)

        chart_predicted_prices = alt.Chart(data_predicted_prices).mark_line().encode(
            x='date:T',
            y='predicted_price:Q',
            color='symbol:N',
            strokeDash='symbol:N',
            tooltip=['symbol', 'date', 'predicted_price'],
        )

        points_predicted_prices = chart_predicted_prices.mark_circle().encode(
            opacity=alt.value(0)
        ).add_selection(
            highlight_predicted_prices
        )

        lines_predicted_prices = chart_predicted_prices.mark_line().encode(
            size=alt.condition(~highlight_predicted_prices, alt.value(1), alt.value(3))
        )

        layer_predicted_prices = (points_predicted_prices + lines_predicted_prices).interactive()
        st.altair_chart(layer_predicted_prices, use_container_width=True)

        st.write('Showing actual and predicted prices for a company')

        company_name = st.selectbox('Choose company name:', sorted(COMPANY_NAMES_TO_STOCK_NAMES.keys()))
        stock_name = COMPANY_NAMES_TO_STOCK_NAMES[company_name]
        data_predicted_actual_prices = load_data_for_predicted_actual_prices_plot(stock_name)

        nearest_predicted_actual_prices = alt.selection(type='single', nearest=True, on='mouseover',
                                                        fields=['date'], empty='none')

        line = alt.Chart(data_predicted_actual_prices).mark_line(interpolate='basis').encode(
            x='date:T',
            y='price:Q',
            color='price_type:N'
        )

        selectors_predicted_actual_prices = alt.Chart(data_predicted_actual_prices).mark_point().encode(
            x='date:T',
            opacity=alt.value(0),
        ).add_selection(
            nearest_predicted_actual_prices
        )

        points_predicted_actual_prices = line.mark_point().encode(
            opacity=alt.condition(nearest_predicted_actual_prices, alt.value(1), alt.value(0))
        )

        text_predicted_actual_prices = line.mark_text(align='left', dx=10, dy=-10).encode(
            text=alt.condition(nearest_predicted_actual_prices, 'price:Q', alt.value(' '))
        )

        rules_predicted_actual_prices = alt.Chart(data_predicted_actual_prices).mark_rule(color='#f63366').encode(
            x='date:T',
        ).transform_filter(
            nearest_predicted_actual_prices
        )

        layer_predicted_actual_prices = alt.layer(
            line, selectors_predicted_actual_prices, points_predicted_actual_prices,
            rules_predicted_actual_prices, text_predicted_actual_prices
        ).interactive()

        st.altair_chart(layer_predicted_actual_prices, use_container_width=True)


if __name__ == "__main__":
    main()
