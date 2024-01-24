import base64
import altair as alt
import streamlit as st

from pandas.plotting import register_matplotlib_converters
from pathlib import Path
from PIL import Image
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

register_matplotlib_converters()


# Configuration de l'app (html, java script like venv\)

# Deploy the app localy in terminal: streamlit run model.py

st.set_page_config(
    page_title="Introduction to Data Science", layout="wide", page_icon="./images/flask.png"
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

st.image("images/AI.jpg")

st.title("Introduction to Data Science")
st.subheader("HEC Executive MBA 2023-2024")
st.markdown("Course provided by **Shirish C. SRIVASTAVA**")

# Image HEC
# image_hec = Image.open('images/hec.png')
# st.image(image_hec, width=300)

st.markdown("---")

st.markdown("(content to be added)")

# default text for st.text_area()

#st.image("images/AI.jpg")




##################################################################################
############################# DASHBOARD PART #####################################
##################################################################################


from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page("main_page.py", "Home", "🏠"),
        Page("pages/timeseries_analysis.py", "Time Series Forecasting", "📈"),
        Page("pages/sentiment_analysis.py", "Sentiment Analysis", "🤔"),
        #Page("pages/image_classification.py", "Image classification", ":camera:"),
        #Page("pages/object_detection.py", "Object Detection", "📹"), #need to reduce RAM costs
        Page("pages/recommendation_system.py", "Recommendation system", "🛒")
    ]
)


## Hi! PARIS logo
st.markdown("  ")
image_hiparis = Image.open('images/hi-paris.png')
st.image(image_hiparis, width=150)
url = "https://www.hi-paris.fr/"
st.markdown("**The app was made in collaboration with: [Hi! PARIS Engineering Team](%s)**" % url)





