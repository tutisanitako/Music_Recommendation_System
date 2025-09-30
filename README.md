# Song Recommender

This project is a simple song recommendation system using track data. It recommends similar songs based on audio features and playlist genre using cosine similarity.

## Features

- Combines numerical audio features and playlist genre text data.
- Uses cosine similarity to find similar songs.
- Saves fitted vectorizers and similarity results with `joblib` for faster re-runs.
- Visualizes distributions of features and popularity.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
