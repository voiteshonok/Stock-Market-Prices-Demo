import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly as py
import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import numpy as np
import altair as alt
import datetime

def get_data_frame_from_tigger(ETF_NAME):
    df = pd.read_csv(ETF_NAME.lower() + '.us.txt', sep=',')
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= datetime.datetime(2010, 1, 1))]
    df = df[(df["Date"] <= datetime.datetime(2017, 12, 31))]
    df.set_index(pd.Series(range(0, len(df))), inplace=True)
    return df

def RSI(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def stochastic(df, k, d):
    df = df.copy()
    low_min = df['Low'].rolling(window=k).min()
    high_max = df['High'].rolling(window=k).max()
    df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=d).mean()
    return df


def main():
    st.title("Stock Market Prices Demo")
    nav = st.sidebar.radio("Navigation", ["Introduction", "Feature Engineering", "Prediction"])

    ETF_NAME = 'SPY'
    df = get_data_frame_from_tigger(ETF_NAME)

    if nav == "Introduction":
        width = 750
        height = 500

        st.header("Introduction")

        st.subheader("Description of the business task")
        st.write("The stock market is known as a place where people can make a fortune if they can crack\n"
                "the mantra to successfully predict stock prices. Though it’s impossible to predict\n"
                "a stock price correctly most the time. So, the question arises, if humans can estimate\n"
                "and consider all factors to predict a movement or a future value of a stock,\n"
                "why can’t machines? Or, rephrasing, how can we make machines predict the value\n"
                "for a stock? Scientists, analysts, and researchers all over the world have been trying\n"
                "to devise a way to answer these questions for a long time now.")

        st.subheader("Dataset")
        st.write("We will be using [Huge Stock Market Dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs).\n"
                "Thera are a lot of companies in this dataset, but we will use data for CERN, Yandex, ect...")
        st.write("**_Context_**\n\n"
                "High-quality financial data is expensive to acquire and is therefore rarely shared for free.\n"
                "There is provided the full historical daily price and volume data for all US-based stocks\n"
                "and ETFs trading on the NYSE, NASDAQ, and NYSE MKT. It's one of the best datasets of its kind\n"
                "you can obtain.")
        st.write("**_Content_**\n\n"
                "The data is presented in CSV format as follows: Date, Open, High, Low,\n"
                "Close, Volume, OpenInt. We will train model on data from 2010 to 2016 years,\n"
                "prediction will be on 2017 year. Note that prices have been adjusted for dividends and splits.\n"
                "To demonstrate data will select CERN. There you can see head of dataset:")
        st.table(df.head())
        st.write("Let's analyze the description. This is the structure. It has ‘Date’ as the index feature.\n"
                "‘High’ denotes the highest value of the day. ‘Low’ denotes the lowest. ‘Open’ is the opening\n"
                "Price and ‘Close’ is the closing for that Date. Now, sometimes close values are regulated\n"
                "by the companies. So the final value is the ‘Adj Close’ which is the same as ‘Close’ Value\n"
                "if the stock price is not regulated. ‘Volume’ is the amount of Stock of that company traded\n"
                "on that date.")

        st.subheader("Plotting dataset")
        st.write("On the chart below you can see you can watch how stock prices for CERN have changed in 2010.")
        df_intro = df[["Date", "Open", "High", "Low", "Close"]]
        df_intro = df_intro[(df["Date"] <= datetime.datetime(2011, 12, 31))]
        df_intro.set_index("Date", inplace=True)
        st.line_chart(df_intro)
        
    if nav == "Feature Engineering":
        width = 750
        height = 500

        st.header("Feature Engineering")

        st.subheader("Historical ETF prices")
        st.text("Data frame with historial prices for SPY fund consists of 3201 rows, each with 7 columns\n"
                "which are: Date, Open/High/Low/Close prices, Volume count and Open Interest number. \n"
                "OpenInt column has only 0 values, so I will just ignore it and focus on the rest\n"
                "of information.In tables below you can see sample prices from the data frame and\n"
                "also few statistics about each column e.g. min/max values, standard deviation etc.")

        if st.checkbox("Show Head"):
            st.table(df.head())

        if st.checkbox("Show Description"):
            st.table(df.describe())

        st.subheader("I. OHLC Chart")
        st.text("An OHLC chart shows the open, high, low and close prices of a stock. It shows you how\n"
                "the price was changing during a particular day and give you a sense of e.g. momentum or\n"
                "volatility of stock. The tip of the lines represent the low and high values and\n"
                "the horizontal segments represent the open and close values. Sample points where\n"
                "the close value is higher (lower) then the open value are called increasing (decreasing).\n"
                "By default, increasing items are drawn in green whereas decreasing are drawn in red.")

        fig = go.Figure([go.Ohlc(x=df.Date,
                                 open=df.Open,
                                 high=df.High,
                                 low=df.Low,
                                 close=df.Close)])
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("II. Volume")
        st.text("A volume is a very basic measure that shows a number of shares traded (bought, sold) over\n"
                "a certain period of time e.g. daily. It is such a simple but often overlooked indicator.\n"
                "Volume is so important because it basically represents the activity in stock trading.\n"
                "Higher volume value indicates higher interests in trading a stock.")

        fig = go.Figure(go.Bar(x=df.Date, y=df.Volume, name='Volume', marker_color='red'))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("III. Moving Averages")
        st.text("Moving Averages (MA) help to smooth out stock prices on a chart by filtering out\n"
                "short-term price fluctuations. We calculate moving averages over a defined period\n"
                "of time e.g. last 9, 50 or 200 days. There are two (most common) averages used in\n"
                "technical analysis which are:\n"
                "   •Simple Moving Average (SMA) - a simple average calculated over last N days\n"
                "e.g. 50, 100 or 200\n"
                "   •Exponential Moving Average (EMA) - an average where greater weights\n"
                "are applied to recent prices")

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
        st.text("Another commonly used indicator is a Relative Strength Index (RSI) that indicates\n"
                "magnitude of recent price changes. It can show that a stock is either\n"
                "overbought or oversold. Typically RSI value of 70 and above signal that a stock\n"
                "is becoming overbought/overvalued, meanwhile value of 30 and less can mean\n"
                "that it is oversold. Full range of RSI is from 0 to 100.")

        num_days = 365
        df['RSI'] = RSI(df).fillna(0)
        fig = go.Figure(go.Scatter(x=df.Date.tail(num_days), y=df.RSI.tail(num_days)))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("V. MACDI")
        st.text("Moving Average Convergence Divergence (MACD) is an indicator which shows\n"
                "the relationship between two exponential moving averages i.e. 12-day and\n"
                "26-day EMAs. We obtain MACD by substracting 26-day EMA (also called slow EMA)\n"
                "from the 12-day EMA (or fast EMA).")

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
        st.text("The last technical tool in this notebook is a stochastic oscillator\n"
                "is quite similar to RSI in the sense that it's values (also in range 0-100)\n"
                "can indicate whether a stock is overbought/oversold or not. It is arguably\n"
                "the most complicated indicator compared to the ones introduced earlier.\n"
                "Stochastic can be calculated as:\n")
        # Todo: Here must be latex

        stochs = stochastic(df, k=14, d=3)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_k.tail(365), name='K stochastic'))
        fig.add_trace(go.Scatter(x=df.Date.tail(365), y=stochs.stoch_d.tail(365), name='D stochastic'))
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_layout(width=width, height=height)
        st.plotly_chart(fig, use_container_width=True)

    if nav == "Prediction":
        @st.cache
        def get_UN_data():
            AWS_BUCKET_URL = "https://streamlit-demo-data.s3-us-west-2.amazonaws.com"
            df = pd.read_csv(AWS_BUCKET_URL + "/agri.csv.gz")
            return df.set_index("Region")

        try:
            df = get_UN_data()
            countries = st.multiselect(
                "Choose countries", list(df.index), ["China", "United States of America"]
            )
            if not countries:
                st.error("Please select at least one country.")
            else:
                data = df.loc[countries]
                data /= 1000000.0
                st.write("### Gross Agricultural Production ($B)", data.sort_index())

                data = data.T.reset_index()
                data = pd.melt(data, id_vars=["index"]).rename(
                    columns={"index": "year", "value": "Gross Agricultural Product ($B)"}
                )
                chart = (
                    alt.Chart(data)
                        .mark_area(opacity=0.3)
                        .encode(
                        x="year:T",
                        y=alt.Y("Gross Agricultural Product ($B):Q", stack=None),
                        color="Region:N",
                    )
                )
                st.altair_chart(chart, use_container_width=True)
        except urllib.error.URLError as e:
            st.error(
                """
                **This demo requires internet access.**

                Connection error: %s
            """
                % e.reason
            )


if __name__ == "__main__":
    main()
