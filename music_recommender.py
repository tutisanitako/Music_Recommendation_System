import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

tracks_high_popularity = pd.read_csv('data/high_popularity_spotify_data.csv')
tracks_low_popularity = pd.read_csv('data/low_popularity_spotify_data.csv')

tracks = pd.concat([tracks_high_popularity, tracks_low_popularity], ignore_index=True)

features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness', 'instrumentalness', 'acousticness', 'duration_ms']

try:
    song_vectorizer = joblib.load('song_vectorizer.pkl')
    print("Loaded pre-existing song_vectorizer.pkl")
except FileNotFoundError:
    song_vectorizer = CountVectorizer()
    song_vectorizer.fit(tracks['playlist_genre'])
    joblib.dump(song_vectorizer, 'song_vectorizer.pkl')
    print("Fitted and saved song_vectorizer.pkl")

tracks.dropna(inplace=True)
tracks = tracks.drop(['id', 'track_href', 'uri', 'analysis_url', 'track_id', 'track_album_id', 'playlist_id', 'track_album_release_date'], axis=1)

tracks = tracks.sort_values(by=['track_popularity'], ascending=False)
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)

floats = [col for col in tracks.columns if tracks[col].dtype == 'float']

num_cols = 4
num_rows = (len(floats) // num_cols) + (1 if len(floats) % num_cols != 0 else 0)

plt.subplots(figsize=(15, num_rows * 3))
for i, col in enumerate(floats):
    plt.subplot(num_rows, num_cols, i + 1)
    sb.histplot(tracks[col], kde=True)
    plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
genre_popularity = tracks.groupby('playlist_genre')['track_popularity'].mean().sort_values(ascending=False)
genre_popularity.plot(kind='bar', color='skyblue')
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sb.histplot(tracks['track_popularity'], kde=True, color='orange')
plt.title('Distribution of Track Popularity')
plt.xlabel('Track Popularity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

tracks = tracks.sort_values(by=['track_popularity'], ascending=False).head(10000)

def get_similarities(song_name, data):
    song_data = data[data['track_name'] == song_name]
    text_array1 = song_vectorizer.transform(song_data['playlist_genre']).toarray()
    num_array1 = song_data[features].to_numpy()

    num_array1 = num_array1.reshape(1, -1)

    similarities = []
    for idx, row in data.iterrows():
        song = row['track_name']
        text_array2 = song_vectorizer.transform([row['playlist_genre']]).toarray()
        num_array2 = row[features].to_numpy()

        num_array2 = num_array2.reshape(1, -1)

        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        similarities.append(text_sim + num_sim)

    return similarities

def recommend_songs(song_name, data=tracks):
    song_data = tracks[tracks['track_name'] == song_name]
    if song_data.shape[0] == 0:
        print(f"'{song_name}' not found. Here are some other songs you may like:")
        for song in data.sample(n=5)['track_name'].values:
            print(song)
        return

    song_artist = song_data['track_artist'].values[0]

    try:
        data = joblib.load('song_similarities.pkl')
        print("Loaded pre-existing song_similarities.pkl")
    except FileNotFoundError:
        data['similarity_factor'] = get_similarities(song_name, data)
        joblib.dump(data, 'song_similarities.pkl')
        print("Calculated and saved song_similarities.pkl")

    data.sort_values(by=['similarity_factor', 'track_popularity'], ascending=[False, False], inplace=True)

    print(f"Recommended songs based on '{song_name}' by {song_artist}:")
    print(data[['track_name', 'track_artist']].iloc[2:7])

recommend_songs('Fear of the Dark - 2015 Remaster')
