import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error 


## logo and url name
st.set_page_config(
    page_title="Spotify Statistics🎵",
    layout="centered",
    page_icon="🎵",
)

##upload data
df = pd.read_csv(
    "Most Streamed Spotify Songs 2024.csv",
    encoding="latin-1", 
    thousands=","
)

##title page
st.title("Music Top Chart Statistics🎵")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Prediction", "Conclusion"])
st.image("music-note-image-5.png")

##making sidebar green
st.markdown("""<style>[data-testid="stSidebar"] { background-color: #1DB954;} </style>""", unsafe_allow_html=True)


##creating introduction page
if page == "Introduction":

    st.subheader("01 Introduction 📈")
   
    ##project overview write up
    st.write("""
    ### 
    Over 70 percent of the world's population listens to music every single day. 
    By looking at the 2024 Top Charts for music across 12 streaming platforms, this project 
    examines the modern trends and hidden patterns driving global song popularity.
             
    ### Research Question
    Can playlist exposure and social media engagement predict how many streams
    a song will receive on Spotify?

    ### Objectives
    - Explore factors associated with highly streamed songs
    - Visualize relationships between playlists, reach, and streams
    - Identify which variables are most strongly related to success
    - Build a linear regression model that predicts Spotify stream count
    """)


    st.markdown("<h3 style='color:#1DB954;'>Data Preview</h3>",unsafe_allow_html=True)
    ##user inputs how many top songs to see. MIN = 1, MAX 100, DEFAULT = 10
    st.markdown("##### Top Songs")
    num_songs = st.slider("Select a number of top songs to display", 1, 1000, 10)
    st.dataframe(df.head(num_songs))

    ##checks for missing values in 'Album Name" row and reports how many
    st.markdown("<h3 style='color:#1DB954;'>Missing Values</h3>",unsafe_allow_html=True)
    st.markdown("##### Songs without albums")
    missing = df["Album Name"].isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("Every selected Top Song is on an album")
    else:
        st.warning("Some top songs are not on albums")

    ##
    st.markdown("<h3 style='color:#1DB954;'>Song Statistics</h3>",unsafe_allow_html=True)
    if st.button("Click to show Describe Table for song popularity"):
        st.dataframe(df.describe())



##visualization page
if page == "Data Visualization":

    st.subheader("02 Data Visualization🧑‍🎤")

    ##bar chart of streams on each streaming platform for top songs
    st.markdown("<h3 style='color:#1DB954;'>Total Streams of all songs by Platform</h3>",unsafe_allow_html=True)

    platforms = [
        "Spotify Streams", 
        "YouTube Views", 
        "TikTok Views",
        "Apple Music Playlist Count", 
        "Amazon Playlist Count", 
        "Pandora Streams"
    ]
    
    totals = [
        df["Spotify Streams"].sum(),
        df["YouTube Views"].sum(),
        df["TikTok Views"].sum(),
        df["Apple Music Playlist Count"].sum(),
        df["Amazon Playlist Count"].sum(),
        df["Pandora Streams"].sum()
    ]
    chart_df = pd.DataFrame({"Total Count": totals}, index=platforms)
    st.bar_chart(chart_df, height=500, color="#1DB954")
    st.markdown("Most streams come from Spotify and Tiktok (social media")

    ##pie chart of songs that are explicit or not
    explicit = (df["Explicit Track"] == 1).sum()
    non_explicit = (df["Explicit Track"] == 0).sum()
    labels = ["Explicit", "Non-Explicit"]
    sizes = [explicit, non_explicit]
    fig, ax = plt.subplots()
    ax.pie(sizes,labels=labels,autopct="%1.1f%%")
    st.markdown("<h3 style='color:#1DB954;'>Explicit vs Non-Explicit Songs</h3>",unsafe_allow_html=True)

    st.pyplot(fig)
    st.markdown("There are more Non-explicit songs on the Top Charts")

    ##lollipop chart of top artists 
    st.markdown("<h3 style='color:#1DB954;'>Top Artists on the Charts</h3>", unsafe_allow_html=True)

    artist_column = "Artist"
    if artist_column in df.columns:
        top_artists = df[artist_column].value_counts().head(10)
    fig_lolly, ax_lolly = plt.subplots(figsize=(10, 6))  

    ax_lolly.hlines(
        y=top_artists.index, 
        xmin=0, 
        xmax=top_artists.values, 
        color='#191414',      
        linewidth=2
    )
    ax_lolly.plot(
        top_artists.values, 
        top_artists.index, 
        "o", 
        color='#1DB954',       
        markersize=12
    )
    for index, value in enumerate(top_artists.values):
        ax_lolly.text(value + 0.2, index, str(value), va='center', fontweight='bold')
    st.pyplot(fig_lolly)
    st.write("These Artists have a higher percentage of their songs being on the Top Charts")

    ## User is able to make their own charts
    st.markdown("<h3 style='color:#1DB954;'>Choose your own x and y value and plot type to make your own graph!</h3>", unsafe_allow_html=True)

    col_x = st.selectbox("Select X-axis variable", df.columns, index=0)
    col_y = st.selectbox("Select Y-axis variable", df.columns, index=1)
    tab1, tab2, tab3 = st.tabs(["Bar Chart 📊", "Line Chart 📈", "Correlation Heatmap 🔥"])

    with tab1:
        st.subheader("Bar Chart")
        sorted_df = df.sort_values(by=col_x)
        st.bar_chart(data=sorted_df, x=col_x, y=col_y, use_container_width=True)

    with tab2:
        st.subheader("Line Chart")
        sorted_df = df.sort_values(by=col_x)
        st.line_chart(data=sorted_df, x=col_x, y=col_y, use_container_width=True)

    with tab3:
        st.subheader("Correlation Matrix")
        df_numeric = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10)) 
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    
        plt.tight_layout()
        st.pyplot(fig_corr)

## prediction page
if page == "Prediction":

    st.subheader("03 Prediction 🔮")
    st.write("""
    This page uses a Linear Regression model to predict Spotify Streams based on
    playlist exposure and social media engagement.
    """)

    # clean numeric columns
    columns_needed = [
        "Spotify Streams",
        "Spotify Playlist Count",
        "Spotify Playlist Reach",
        "YouTube Views",
        "TikTok Views",
        "Apple Music Playlist Count",
        "Amazon Playlist Count",
        "Pandora Streams"
    ]

    model_df = df[columns_needed].copy()

    for col in columns_needed:
        model_df[col] = (
            model_df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
        )
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.dropna()

    X = model_df[
        [
            "Spotify Playlist Count",
            "Spotify Playlist Reach",
            "YouTube Views",
            "TikTok Views",
            "Apple Music Playlist Count",
            "Amazon Playlist Count",
            "Pandora Streams"
        ]
    ]

    y = model_df["Spotify Streams"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.markdown("<h3 style='color:#1DB954;'>Model Results</h3>", unsafe_allow_html=True)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("R² Score:", r2)
    st.write("Mean Absolute Error:", mae)

    st.write("""
    The R² score shows how well the model explains variation in Spotify streams.
    A higher R² means the prediction model is stronger.
    """)

    # Linear regression graph
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Spotify Streams")
    ax.set_ylabel("Predicted Spotify Streams")
    ax.set_title("Actual vs Predicted Spotify Streams")

    st.pyplot(fig)

    st.markdown("<h3 style='color:#1DB954;'>Make Your Own Prediction</h3>", unsafe_allow_html=True)

    spotify_playlist_count = st.number_input("Spotify Playlist Count", min_value=0)
    spotify_playlist_reach = st.number_input("Spotify Playlist Reach", min_value=0)
    youtube_views = st.number_input("YouTube Views", min_value=0)
    tiktok_views = st.number_input("TikTok Views", min_value=0)
    apple_playlist_count = st.number_input("Apple Music Playlist Count", min_value=0)
    amazon_playlist_count = st.number_input("Amazon Playlist Count", min_value=0)
    pandora_streams = st.number_input("Pandora Streams", min_value=0)

    user_data = [[
        spotify_playlist_count,
        spotify_playlist_reach,
        youtube_views,
        tiktok_views,
        apple_playlist_count,
        amazon_playlist_count,
        pandora_streams
    ]]

    if st.button("Predict Spotify Streams"):
        prediction = model.predict(user_data)
        st.success(f"Predicted Spotify Streams: {prediction[0]:,.0f}")

## conclusion page
if page == "Conclusion":

    st.subheader("04 Conclusion 🎵")

    st.write("""
    This project explored factors that influence song popularity on Spotify.
    Through data visualization and linear regression modeling, we found that
    playlist exposure, social media engagement, and platform reach are strongly
    associated with higher Spotify stream counts.

    The regression model demonstrated that these factors can be used to predict
    the popularity of a song, helping artists and music companies better
    understand what contributes to streaming success.

    Overall, songs with greater playlist presence and online engagement tend to
    receive more streams, showing the importance of digital promotion in the
    modern music industry.
    """)

    st.success("Project Complete! 🎉")
