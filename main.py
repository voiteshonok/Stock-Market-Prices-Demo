import urllib

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import time
import numpy as np
import altair as alt


# Hello World


def main():
    st.title("Salary Predictor")

    nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])
    if nav == "Introduction":

        if st.checkbox("Show Table"):
            st.table(data)

        graph = st.selectbox("What kind of Graph ? ", ["Non-Interactive", "Interactive"])

        val = st.slider("Filter data using years", 0, 20)
        if graph == "Non-Interactive":
            plt.figure(figsize=(10, 5))
            plt.ylim(0)
            plt.xlabel("Years of Experience")
            plt.ylabel("Salary")
            plt.tight_layout()
            st.pyplot()

    if nav == "Feature Engineering":
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = np.random.randn(1, 1)
        chart = st.line_chart(last_rows)

        for i in range(1, 101):
            new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            chart.add_rows(new_rows)
            progress_bar.progress(i)
            last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()

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
