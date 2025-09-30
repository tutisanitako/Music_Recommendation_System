import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
tracks_high_popularity = pd.read_csv('data/high_popularity_spotify_data.csv')
tracks_low_popularity = pd.read_csv('data/low_popularity_spotify_data.csv')

# Combine both datasets
tracks = pd.concat([tracks_high_popularity, tracks_low_popularity], ignore_index=True)

# Features to be used for similarity calculation
features = ['energy', 'tempo', 'danceability', 'loudness', 'liveness', 'valence', 'speechiness', 'instrumentalness', 'acousticness', 'duration_ms']

# Initialize the CountVectorizer
song_vectorizer = CountVectorizer()

# Fit the vectorizer on the playlist genre data
song_vectorizer.fit(tracks['playlist_genre'])

# Data Cleaning
tracks.dropna(inplace=True)  # Remove rows with missing values
tracks = tracks.drop(['id', 'track_href', 'uri', 'analysis_url', 'track_id', 'track_album_id', 'playlist_id', 'track_album_release_date'], axis=1)

# Remove duplicate songs based on track name
tracks = tracks.sort_values(by=['track_popularity'], ascending=False)
tracks.drop_duplicates(subset=['track_name'], keep='first', inplace=True)

# Identify columns with float data type
floats = [col for col in tracks.columns if tracks[col].dtype == 'float']

# Set the number of columns and rows for subplots
num_cols = 4  # Reduce the number of columns to 4
num_rows = (len(floats) // num_cols) + (1 if len(floats) % num_cols != 0 else 0)

# Plotting the distribution of continuous features
plt.subplots(figsize=(15, num_rows * 3))  # Adjust height to fit better
for i, col in enumerate(floats):
    plt.subplot(num_rows, num_cols, i + 1)  # Dynamically create subplots
    sb.histplot(tracks[col], kde=True)  # Use histplot instead of distplot for newer seaborn versions
    plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

# Example: Average Popularity by Genre (Bar chart)
plt.figure(figsize=(12, 6))
genre_popularity = tracks.groupby('playlist_genre')['track_popularity'].mean().sort_values(ascending=False)
genre_popularity.plot(kind='bar', color='skyblue')
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Example: Distribution of Track Popularity (Histogram)
plt.figure(figsize=(8, 6))
sb.histplot(tracks['track_popularity'], kde=True, color='orange')
plt.title('Distribution of Track Popularity')
plt.xlabel('Track Popularity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Sort by popularity and select the top 10,000 songs
tracks = tracks.sort_values(by=['track_popularity'], ascending=False).head(10000)

# Function to calculate similarities
def get_similarities(song_name, data):
    # Get the vector for the input song
    song_data = data[data['track_name'] == song_name]
    text_array1 = song_vectorizer.transform(song_data['playlist_genre']).toarray()
    num_array1 = song_data[features].to_numpy()

    # Reshape num_array1 to 2D
    num_array1 = num_array1.reshape(1, -1)

    # Calculate similarities for all songs
    similarities = []
    for idx, row in data.iterrows():
        song = row['track_name']
        text_array2 = song_vectorizer.transform([row['playlist_genre']]).toarray()
        num_array2 = row[features].to_numpy()

        # Reshape num_array2 to 2D
        num_array2 = num_array2.reshape(1, -1)

        # Calculate cosine similarity for text and numerical features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        similarities.append(text_sim + num_sim)

    return similarities

# Function to recommend songs
def recommend_songs(song_name, data=tracks):
    # Check if the song exists in the dataset
    song_data = tracks[tracks['track_name'] == song_name]
    if song_data.shape[0] == 0:
        print(f"'{song_name}' not found. Here are some other songs you may like:")
        for song in data.sample(n=5)['track_name'].values:
            print(song)
        return

    # Get the artist of the input song
    song_artist = song_data['track_artist'].values[0]

    # Calculate similarity for the input song
    data['similarity_factor'] = get_similarities(song_name, data)

    # Sort by similarity and popularity
    data.sort_values(by=['similarity_factor', 'track_popularity'], ascending=[False, False], inplace=True)

    # Display recommended songs (excluding the input song itself)
    print(f"Recommended songs based on '{song_name}' by {song_artist}:")
    print(data[['track_name', 'track_artist']].iloc[2:7])  # Skipping the first song as it's the input song

# Example usage of the recommender system
recommend_songs('Fear of the Dark - 2015 Remaster')
