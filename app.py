import streamlit as st
from prediction import prediction_function
from sentiment import sentiment_function

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    layout="wide",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
column1, column2 = st.columns(2)
with column1:
  prediction_function()
with column2:
  sentiment_function()
