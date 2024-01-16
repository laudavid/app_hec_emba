import pandas as pd
from pandas import to_datetime
from pandas.plotting import register_matplotlib_converters
import numpy as np
from pathlib import Path
import base64
import io
import os
#import yfinance as yf


import altair as alt
from PIL import Image
import streamlit as st


from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import seaborn as sns
import matplotlib.pyplot as plt
import re
register_matplotlib_converters()

import scipy.optimize as opt
import yfinance as yf
import datetime
import time
import glob
import os
import itertools
import matplotlib.image as mpimg

#from numba import jit
from scipy.stats import norm
from sklearn.metrics import mean_squared_error as mse



sns.set(style="whitegrid")
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
st.set_option('deprecation.showPyplotGlobalUse', False)






# Configuration de l'app (html, java script like venv\)

# Deploy the app localy in terminal: streamlit run model.py

st.set_page_config(
    page_title="Finance", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

@st.cache_data # compression data
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source


@st.cache_data
def get_chart(data):
    hover = alt.selection_single(
        fields=["Date_2"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="Date_2",
            y=kpi,
            #color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y=kpi,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip(kpi, title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


@st.cache_data
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# from st_files_connection import FilesConnection

# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.experimental_connection('gcs', type=FilesConnection)
# df = conn.read("streamlit-hiparis/master.csv", input_format="csv", ttl=600)
# st.dataframe(df)


# Image Hi Paris
image_hiparis = Image.open('images/hi-paris.png')




##################################################################################
#################################### PASSWORD ####################################
##################################################################################

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" in st.session_state and st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


# if check_password():
#     # st.write("Here goes your normal Streamlit app...")
#     # st.button("Click me")



# ########### TITLE #############

# Image HEC
image_hec = Image.open('images/hec.png')
st.image(image_hec, width=300)

st.title("Introduction to Data Science")
st.subheader("HEC EMBA 2023-2024")
st.markdown("Course provided by **Shirish C. SRIVASTAVA**")

st.markdown("  ")
st.markdown("---")

# default text for st.text_area()
default_text = ""




##################################################################################
############################# DASHBOARD PART #####################################
##################################################################################

# st.sidebar.image(image_hiparis, width=200)
# url = "https://www.hi-paris.fr/"
# st.sidebar.markdown("Made in collaboration with the [Hi! PARIS Engineering Team](%s)" % url)

#st.sidebar.markdown("  ")



# st.sidebar.header("**Dashboard**") # .sidebar => add widget to sidebar
# st.sidebar.markdown("  ")
#st.markdown("  ")
#st.sidebar.divider()
#st.sidebar.markdown("---")

############# OPEN CHATBOT (Langchain OpenAI) #############
# if st.sidebar.button('**Open Chatbot**'):
#     st.write("Hello")


#st.sidebar.button("My student id isn't in the list")

# Select lab/exercice number
# select_page = st.sidebar.selectbox('Select the section ➡️', [
# 'Introduction',
# '01 - Time Series Forecasting',
# '02 - Object Detection', 
# '03 - Audio Sentiment Analysis'
# #   '03 - Diversification',
# #   '04 - Test of the CAPM',
# ])


from st_pages import Page, show_pages, add_page_title

# Optional -- adds the title and icon to the current page

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("model.py", "Home", "🏠"),
        Page("pages/ts_analysis.py", "Time Series Analysis", "📈"),
        Page("pages/sentiment_analysis.py", "Sentiment Analysis", "🤔"),
        Page("pages/object_detection.py", "Object Detection", ":camera:"),
        Page("pages/recommendation_system.py", "Recommendation system", "🛒")
    ]
)



##################################################################################
#               USE CASE 1: Time Series Forecasting of Stock Volatility
##################################################################################



if __name__=='__main__':
    main()

#st.markdown(" ")
#st.markdown("### 👨🏼‍💻 **App Contributors:** ")
#st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

#st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/app-predictive-analytics'}) 🚀 ")
#st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made in collaboration with: ",
        link("https://www.hi-paris.fr/", "Hi! PARIS Engineering Team"),
        "👨🏼‍💻"
    ]
    layout(*myargs)


# if __name__ == "__main__":
#     footer2()

