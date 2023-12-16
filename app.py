import streamlit as st
from prediction import prediction_function
from sentiment import sentiment_function


column1, column2 = st.columns(2)
with column1:
  prediction_function()
with column2:
  sentiment_function()