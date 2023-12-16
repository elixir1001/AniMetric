import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
def sentiment_function():
    st.markdown(
        """
        <div style='background: linear-gradient(to right, #355834, #355834); padding: 1px; border-radius: 0px; text-align: center;'>
            <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 26px;'>Japanese Seasonal Animation Toxicity Meter</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='background: linear-gradient(to right, #355834, #355834); padding: 1px; border-radius: 0px; text-align: center;'>
            <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 26px;'>日本の季節アニメ毒性メーター</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write('---')

    myanimelist_url = st.text_input('Enter the link of MAL for the animation', 'https://myanimelist.net/anime/')

    r = requests.get(myanimelist_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*text.*')

    results = soup.find_all('div', {'class':regex})
    reviews = [result.text for result in results]

    df = pd.DataFrame(np.array(reviews), columns=['review'])

    from textblob import TextBlob

    # Assuming 'df' is your dataframe and 'column_name' is the name of the column containing strings
    # Replace 'column_name' with the actual column name in your dataframe
    sentiment_scores = []
    for text in df['review']:
        blob = TextBlob(str(text))
        sentiment_scores.append(blob.sentiment.polarity)

    # Adding sentiment scores to the dataframe
    df['sentiment_score'] = sentiment_scores


    average_sentiment = df['sentiment_score'].mean()

    


    name = soup.find('strong')
    if name:
        content_inside_strong = name.text
    st.title(content_inside_strong)

    description_tag = soup.find('p', itemprop='description')  # Find the <p> tag with itemprop='description'

    if description_tag:
        description_text = description_tag.get_text(separator='\n\n')  # Extract the text within the <p> tag
        
        st.write(description_text)  # This will print the extracted description
    else:
        st.write("No <p> tag found with itemprop='description'")


    def label_sentiment(score):
        if score > 0.6:
            return 'Extremely Positive'
        elif 0.4 < score <= 0.6:
            return 'Very Positive'
        elif 0.2 < score <= 0.4:
            return 'Positive'
        elif -0.2 <= score <= 0.2:
            return 'Neutral'
        elif -0.4 <= score < -0.2:
            return 'Negative'
        elif -0.6 <= score < -0.4:
            return 'Very Negative'
        else:
            return 'Extremely Negative'

    # Label the average sentiment score
    average_sentiment_label = label_sentiment(average_sentiment)

    # Display the labeled average sentiment to the user
    st.subheader("Sentiment Analysis of Community: ")
    st.subheader(average_sentiment_label)


    
