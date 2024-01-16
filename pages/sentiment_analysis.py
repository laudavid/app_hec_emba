import os
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.header("Sentiment Analysis (Text, Audio)")

st.markdown("**What is Sentiment Analysis ?**")
st.markdown("""
    Sentiment analysis is a Natural Language Processing (NLP) task that involves determining the sentiment or emotion expressed in a piece of text. 
    It has a wide range of use cases across various industries, as it helps organizations gain insights into the opinions, emotions, and attitudes expressed in text data. 
            
Here are a few examples where Sentiment Analysis can be useful:
- **Customer Feedback and Reviews** 💯: Assessing reviews on products or services to understand customer satisfaction and identify areas for improvement.
- **Market Research** 🔍: Analyzing survey responses or online forums to gauge public opinion on products, services, or emerging trends.
- **Financial Market Analysis** 📉: Monitoring financial news, reports, and social media to gauge investor sentiment and predict market trends.
- **Government and Public Policy**: Analyzing public opinion on government policies, initiatives, and political decisions to gauge public sentiment and inform decision-making.        
""")

st.image("images/sentiment_analysis.png")