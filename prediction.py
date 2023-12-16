import joblib  # For loading the .pkl file
from sklearn.ensemble import RandomForestRegressor
import pandas as pd 
import streamlit as st

def prediction_function():
    model = joblib.load('random_forest_regressor.pkl')
    df = pd.read_csv("AnimeList.csv")

    columns_to_remove = ['anime_id',
                        'title', 
                        'title_english', 
                        'title_japanese', 
                        'title_synonyms',
                        'image_url', 'status', 
                        'airing', 
                        'aired_string', 
                        'aired', 
                        'scored_by', 
                        'rank', 
                        'popularity', 
                        'favorites', 
                        'background', 
                        'broadcast', 
                        'related', 
                        'opening_theme', 
                        'ending_theme',
                        'premiered', 
                        'studio', 
                        'producer']
    df = df.drop(columns_to_remove, axis=1)
    df.dropna(subset=['rating'], inplace=True)
    df.drop('licensor', axis=1, inplace=True)
    df.dropna(subset=['genre'], inplace=True)
    all_genre = df['genre'].str.split(', ').explode().unique().tolist()


    def features_from_user():
        predefined_genre = all_genre
        selected_genre = st.multiselect('Select genre', predefined_genre)
        selected_genre_str = ', '.join(selected_genre)
        col1, col2 = st.columns(2)
        with col1:
            type = st.selectbox("Select the type of Animation", pd.unique(df["type"]))
            source = st.selectbox("Select the original source of the animation", pd.unique(df["source"]))
            episodes = st.slider('Number of episodes',
                                df.episodes.min(),
                                df.episodes.max())
        with col2:
            duration = st.selectbox("Select the duration of the animation episode", pd.unique(df["duration"]))
            rating = st.selectbox("Select the rating of the animation", pd.unique(df["rating"]))
            members = st.slider('Number of current members',
                                df.members.min(),
                                df.members.max())
        

        data = {
            'type': type,
            'source': source,
            'episodes': episodes,
            'duration': duration,
            'rating': rating,
            'members': members,
            'genre': selected_genre_str
        }
        features = pd.DataFrame(data, index=[0])
        return features

    st.markdown(
        """
        <div style='background: linear-gradient(to right,  #355834 ,#355834); padding: 1px; border-radius: 0px; text-align: center;'>
            <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 26px;'>Japanese Seasonal Animation Score Prediction</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='background: linear-gradient(to right,  #355834 ,#355834); padding: 1px; border-radius: 0px; text-align: center;'>
            <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 26px;'>日本の季節アニメのスコア予測
    </h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write('---')


    df_for_prediction = features_from_user()
    genres = df_for_prediction['genre'].str.get_dummies(sep=', ')
    df_for_prediction = pd.concat([df_for_prediction, genres], axis=1)
    df_for_prediction.drop('genre', axis=1, inplace=True)
    columns_to_encode = ['type', 'source', 'duration', 'rating']
    encoded_df = pd.get_dummies(df_for_prediction, columns=columns_to_encode)
    df_for_prediction = pd.concat([df_for_prediction, encoded_df], axis=1)
    new_columns_to_drop = ['type', 'source', 'duration', 'rating']
    df_for_prediction = df_for_prediction.drop(new_columns_to_drop, axis=1)
    original_column_list = ['episodes', 'members', 'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police', 'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Vampire', 'Yaoi', 'Yuri', 'episodes', 'members', 'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama', 'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror', 'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police', 'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life', 'Space', 'Sports', 'Super Power', 'Supernatural', 'Thriller', 'Vampire', 'Yaoi', 'Yuri', 'type_Movie', 'type_Music', 'type_ONA', 'type_OVA', 'type_Special', 'type_TV', 'type_Unknown', 'source_4-koma manga', 'source_Book', 'source_Card game', 'source_Digital manga', 'source_Game', 'source_Light novel', 'source_Manga', 'source_Music', 'source_Novel', 'source_Original', 'source_Other', 'source_Picture book', 'source_Radio', 'source_Unknown', 'source_Visual novel', 'source_Web manga', 'duration_1 hr.', 'duration_1 hr. 1 min.', 'duration_1 hr. 10 min.', 'duration_1 hr. 11 min.', 'duration_1 hr. 12 min.', 'duration_1 hr. 13 min.', 'duration_1 hr. 14 min.', 'duration_1 hr. 14 min. per ep.', 'duration_1 hr. 15 min.', 'duration_1 hr. 15 min. per ep.', 'duration_1 hr. 16 min.', 'duration_1 hr. 16 min. per ep.', 'duration_1 hr. 17 min.', 'duration_1 hr. 17 min. per ep.', 'duration_1 hr. 18 min.', 'duration_1 hr. 19 min.', 'duration_1 hr. 2 min.', 'duration_1 hr. 2 min. per ep.', 'duration_1 hr. 20 min.', 'duration_1 hr. 21 min.', 'duration_1 hr. 22 min.', 'duration_1 hr. 23 min.', 'duration_1 hr. 24 min.', 'duration_1 hr. 24 min. per ep.', 'duration_1 hr. 25 min.', 'duration_1 hr. 26 min.', 'duration_1 hr. 27 min.', 'duration_1 hr. 28 min.', 'duration_1 hr. 29 min.', 'duration_1 hr. 3 min.', 'duration_1 hr. 3 min. per ep.', 'duration_1 hr. 30 min.', 'duration_1 hr. 30 min. per ep.', 'duration_1 hr. 31 min.', 'duration_1 hr. 32 min.', 'duration_1 hr. 33 min.', 'duration_1 hr. 34 min.', 'duration_1 hr. 35 min.', 'duration_1 hr. 35 min. per ep.', 'duration_1 hr. 36 min.', 'duration_1 hr. 36 min. per ep.', 'duration_1 hr. 37 min.', 'duration_1 hr. 38 min.', 'duration_1 hr. 39 min.', 'duration_1 hr. 4 min.', 'duration_1 hr. 40 min.', 'duration_1 hr. 41 min.', 'duration_1 hr. 42 min.', 'duration_1 hr. 43 min.', 'duration_1 hr. 44 min.', 'duration_1 hr. 45 min.', 'duration_1 hr. 46 min.', 'duration_1 hr. 47 min.', 'duration_1 hr. 48 min.', 'duration_1 hr. 49 min.', 'duration_1 hr. 5 min.', 'duration_1 hr. 5 min. per ep.', 'duration_1 hr. 50 min.', 'duration_1 hr. 51 min.', 'duration_1 hr. 52 min.', 'duration_1 hr. 52 min. per ep.', 'duration_1 hr. 53 min.', 'duration_1 hr. 54 min.', 'duration_1 hr. 55 min.', 'duration_1 hr. 56 min.', 'duration_1 hr. 57 min.', 'duration_1 hr. 58 min.', 'duration_1 hr. 59 min.', 'duration_1 hr. 6 min.', 'duration_1 hr. 7 min.', 'duration_1 hr. 7 min. per ep.', 'duration_1 hr. 8 min.', 'duration_1 hr. 9 min.', 'duration_1 hr. per ep.', 'duration_1 min.', 'duration_1 min. per ep.', 'duration_10 min.', 'duration_10 min. per ep.', 'duration_10 sec.', 'duration_10 sec. per ep.', 'duration_11 min.', 'duration_11 min. per ep.', 'duration_12 min.', 'duration_12 min. per ep.', 'duration_12 sec.', 'duration_12 sec. per ep.', 'duration_13 min.', 'duration_13 min. per ep.', 'duration_13 sec.', 'duration_14 min.', 'duration_14 min. per ep.', 'duration_14 sec.', 'duration_14 sec. per ep.', 'duration_15 min.', 'duration_15 min. per ep.', 'duration_15 sec.', 'duration_15 sec. per ep.', 'duration_16 min.', 'duration_16 min. per ep.', 'duration_16 sec.', 'duration_16 sec. per ep.', 'duration_17 min.', 'duration_17 min. per ep.', 'duration_17 sec.', 'duration_18 min.', 'duration_18 min. per ep.', 'duration_18 sec.', 'duration_19 min.', 'duration_19 min. per ep.', 'duration_19 sec. per ep.', 'duration_2 hr.', 'duration_2 hr. 1 min.', 'duration_2 hr. 10 min.', 'duration_2 hr. 11 min.', 'duration_2 hr. 12 min.', 'duration_2 hr. 14 min.', 'duration_2 hr. 15 min.', 'duration_2 hr. 16 min.', 'duration_2 hr. 17 min.', 'duration_2 hr. 19 min.', 'duration_2 hr. 2 min.', 'duration_2 hr. 20 min.', 'duration_2 hr. 21 min.', 'duration_2 hr. 3 min.', 'duration_2 hr. 30 min.', 'duration_2 hr. 31 min.', 'duration_2 hr. 32 min.', 'duration_2 hr. 33 min.', 'duration_2 hr. 4 min.', 'duration_2 hr. 40 min.', 'duration_2 hr. 41 min.', 'duration_2 hr. 42 min.', 'duration_2 hr. 43 min.', 'duration_2 hr. 47 min.', 'duration_2 hr. 5 min.', 'duration_2 hr. 6 min.', 'duration_2 hr. 7 min.', 'duration_2 hr. 8 min.', 'duration_2 min.', 'duration_2 min. per ep.', 'duration_20 min.', 'duration_20 min. per ep.', 'duration_20 sec.', 'duration_20 sec. per ep.', 'duration_21 min.', 'duration_21 min. per ep.', 'duration_21 sec.', 'duration_21 sec. per ep.', 'duration_22 min.', 'duration_22 min. per ep.', 'duration_22 sec.', 'duration_22 sec. per ep.', 'duration_23 min.', 'duration_23 min. per ep.', 'duration_23 sec.', 'duration_24 min.', 'duration_24 min. per ep.', 'duration_24 sec. per ep.', 'duration_25 min.', 'duration_25 min. per ep.', 'duration_25 sec.', 'duration_25 sec. per ep.', 'duration_26 min.', 'duration_26 min. per ep.', 'duration_26 sec.', 'duration_26 sec. per ep.', 'duration_27 min.', 'duration_27 min. per ep.', 'duration_27 sec.', 'duration_28 min.', 'duration_28 min. per ep.', 'duration_28 sec.', 'duration_29 min.', 'duration_29 min. per ep.', 'duration_29 sec.', 'duration_29 sec. per ep.', 'duration_3 hr. 51 min.', 'duration_3 min.', 'duration_3 min. per ep.', 'duration_30 min.', 'duration_30 min. per ep.', 'duration_30 sec.', 'duration_30 sec. per ep.', 'duration_31 min.', 'duration_31 min. per ep.', 'duration_31 sec.', 'duration_31 sec. per ep.', 'duration_32 min.', 'duration_32 min. per ep.', 'duration_32 sec.', 'duration_32 sec. per ep.', 'duration_33 min.', 'duration_33 min. per ep.', 'duration_33 sec.', 'duration_33 sec. per ep.', 'duration_34 min.', 'duration_34 min. per ep.', 'duration_34 sec.', 'duration_34 sec. per ep.', 'duration_35 min.', 'duration_35 min. per ep.', 'duration_35 sec.', 'duration_35 sec. per ep.', 'duration_36 min.', 'duration_36 min. per ep.', 'duration_36 sec.', 'duration_36 sec. per ep.', 'duration_37 min.', 'duration_37 min. per ep.', 'duration_37 sec.', 'duration_37 sec. per ep.', 'duration_38 min.', 'duration_38 min. per ep.', 'duration_38 sec.', 'duration_38 sec. per ep.', 'duration_39 min.', 'duration_39 min. per ep.', 'duration_39 sec.', 'duration_39 sec. per ep.', 'duration_4 min.', 'duration_4 min. per ep.', 'duration_40 min.', 'duration_40 min. per ep.', 'duration_40 sec.', 'duration_40 sec. per ep.', 'duration_41 min.', 'duration_41 min. per ep.', 'duration_41 sec.', 'duration_41 sec. per ep.', 'duration_42 min.', 'duration_42 min. per ep.', 'duration_42 sec.', 'duration_42 sec. per ep.', 'duration_43 min.', 'duration_43 min. per ep.', 'duration_44 min.', 'duration_44 min. per ep.', 'duration_44 sec.', 'duration_44 sec. per ep.', 'duration_45 min.', 'duration_45 min. per ep.', 'duration_45 sec.', 'duration_45 sec. per ep.', 'duration_46 min.', 'duration_46 min. per ep.', 'duration_46 sec.', 'duration_46 sec. per ep.', 'duration_47 min.', 'duration_47 min. per ep.', 'duration_48 min.', 'duration_48 min. per ep.', 'duration_49 min.', 'duration_49 min. per ep.', 'duration_49 sec.', 'duration_49 sec. per ep.', 'duration_5 min.', 'duration_5 min. per ep.', 'duration_50 min.', 'duration_50 min. per ep.', 'duration_50 sec.', 'duration_51 min.', 'duration_51 min. per ep.', 'duration_51 sec.', 'duration_51 sec. per ep.', 'duration_52 min.', 'duration_52 min. per ep.', 'duration_52 sec.', 'duration_53 min.', 'duration_53 min. per ep.', 'duration_53 sec. per ep.', 'duration_54 min.', 'duration_54 min. per ep.', 'duration_54 sec.', 'duration_54 sec. per ep.', 'duration_55 min.', 'duration_55 min. per ep.', 'duration_55 sec.', 'duration_56 min.', 'duration_56 sec. per ep.', 'duration_57 min.', 'duration_57 min. per ep.', 'duration_57 sec.', 'duration_57 sec. per ep.', 'duration_58 min.', 'duration_58 min. per ep.', 'duration_58 sec.', 'duration_58 sec. per ep.', 'duration_59 min.', 'duration_6 min.', 'duration_6 min. per ep.', 'duration_7 min.', 'duration_7 min. per ep.', 'duration_7 sec.', 'duration_8 min.', 'duration_8 min. per ep.', 'duration_9 min.', 'duration_9 min. per ep.', 'duration_Unknown', 'rating_G - All Ages', 'rating_PG - Children', 'rating_PG-13 - Teens 13 or older', 'rating_R - 17+ (violence & profanity)', 'rating_R+ - Mild Nudity', 'rating_Rx - Hentai']

    new_df = pd.DataFrame(0, index=[0], columns=original_column_list)

    # Update values from df for existing columns
    common_columns = list(set(original_column_list) & set(df_for_prediction.columns))
    new_df[common_columns] = df_for_prediction[common_columns].iloc[0]

    final_input = new_df
    predictions = model.predict(final_input)
    single_prediction = predictions[0][0]

    st.subheader("The Predicted Score is: ")
    st.subheader(single_prediction)
