import datetime
import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt

from PIL import Image

# Load the S&P stock
stocks = '^GSPC'
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2021, 8, 1)

s_p500 = yf.download(stocks, start=start, end = end, interval='1d')
ret = 100 * (s_p500.pct_change()[1:]['Adj Close'])

df_volatility = ret.rolling(5).std().reset_index()
df_volatility.rename({"Adj Close":"Volatility"} ,axis=1, inplace=True)

df_volatility["Date"] = pd.to_datetime(df_volatility["Date"]).dt.date
#st.dataframe(df_volatility)



##################################### TITLE ####################################

st.header("Time Series Forecasting")

st.markdown("#### What is Time Series Forecasting ?")
st.markdown("Time series forecasting models are built to make accurate predictions about future values of a time-dependent variable, taking into account patterns, trends, and seasonality observed in it's historical data.")

image_ts = Image.open('images/ts_patterns.png')
st.image(image_ts, width=800)

st.markdown("    ")
st.markdown("    ")


st.info(""" In this first use case, our goal is to predict the volatility of the S&P stock over the period of 2012-2021. 
            The data was collected through Yahoo Finance.
    """)

st.markdown("    ")
st.markdown("    ")



#################################### GRAPH ###################################

vol_chart = alt.Chart(df_volatility).mark_line().encode(alt.X('Date:T', axis=alt.Axis(format='%b %Y')), y="Volatility").properties(title=f'S&P-500 Stock Price Volatility')
st.altair_chart(vol_chart.interactive(), use_container_width=True)


############################ VOLATILITY FORECASTING #########################

from sklearn.svm import SVR
from scipy.stats import uniform as sp_rand
from sklearn.model_selection import RandomizedSearchCV

# Clean data
# df_volatility = df_volatility.dropna().reset_index(drop=True)
# df_volatility.drop(columns=["Date"], inplace=True)

# returns_svm = ret ** 2
# returns_svm = returns_svm.reset_index()

# returns_svm.drop(columns=["Date"], inplace=True)

# X = pd.concat([df_volatility, returns_svm], axis=1, ignore_index=True)
# st.dataframe(X)

# X = X[4:].copy()
# X = X.reset_index()
# X.drop('index', axis=1, inplace=True)

# # Time Series forecasting with linear kernel
# svr_poly = SVR(kernel='poly', degree=2)
# svr_lin = SVR(kernel='linear')
# svr_rbf = SVR(kernel='rbf')

# n = 252
# para_grid = {'gamma': sp_rand(),
#             'C': sp_rand(),
#             'epsilon': sp_rand()}
# clf = RandomizedSearchCV(svr_lin, para_grid)

# clf.fit(X.iloc[:-n].values,
#         df_volatility.iloc[1:-(n-1)].values.reshape(-1,))
# predict_svr_lin = clf.predict(X.iloc[-n:])

# predict_svr_lin = pd.DataFrame(predict_svr_lin)
# predict_svr_lin.index = ret.iloc[-n:].index

# df_volatility.index = ret.iloc[4:].index

# plt.figure(figsize=(8, 4))
# plt.plot(df_volatility / 100, label='Realized Volatility')
# plt.plot(predict_svr_lin / 100, label='Volatility Prediction-SVR-Linear')
# plt.title('Volatility Prediction with SVR-Linear', fontsize=12)
# plt.legend()
# plt.show()


# st.markdown("   ")
# st.markdown("   ") 
# st.markdown("   ")     
# st.markdown("   ")
# st.markdown("   ") 