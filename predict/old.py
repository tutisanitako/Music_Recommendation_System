import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns


# Step 1: Load and preprocess the data
def load_and_preprocess_data():
    # Load the datasets
    low_popularity_data = pd.read_csv('../data/low_popularity_spotify_data.csv')
    high_popularity_data = pd.read_csv('../data/high_popularity_spotify_data.csv')

    # Standardize column names (lowercase and remove spaces)
    low_popularity_data.columns = low_popularity_data.columns.str.lower()
    high_popularity_data.columns = high_popularity_data.columns.str.lower()

    # Add a target column for popularity (0 for low, 1 for high)
    low_popularity_data['popularity_label'] = 0
    high_popularity_data['popularity_label'] = 1

    # Identify common columns between datasets
    common_columns = list(set(low_popularity_data.columns) & set(high_popularity_data.columns))

    # Merge datasets on common columns
    data = pd.concat([low_popularity_data[common_columns], high_popularity_data[common_columns]], ignore_index=True)

    return data


# Step 2: Visualize the data
def visualize_data(data):
    # Genre Distribution (Bar Chart)
    plt.figure(figsize=(10, 6))
    data['playlist_genre'].value_counts().plot(kind='bar', color='skyblue', title='Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Popularity Label Distribution (Pie Chart)
    plt.figure(figsize=(8, 8))
    data['popularity_label'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'],
                                                 title='Popularity Label Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()

    # Energy vs Danceability (Scatter Plot)
    if 'energy' in data.columns and 'danceability' in data.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(data['energy'], data['danceability'], c=data['popularity_label'], cmap='coolwarm', alpha=0.7)
        plt.colorbar(label='Popularity Label')
        plt.xlabel('Energy')
        plt.ylabel('Danceability')
        plt.title('Energy vs Danceability')
        plt.tight_layout()
        plt.show()


# Step 3: Create and train a model
def train_model(data, features):
    # Select features and target
    X = data[features]
    y = data['popularity_label']

    # Handle missing values by filling with the mean
    X = X.fillna(X.mean())

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nModel Accuracy: {accuracy:.2f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Popularity', 'High Popularity'], yticklabels=['Low Popularity', 'High Popularity'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    # Save the trained model for future use
    joblib.dump(model, 'prediction_model.pkl')

    return model


# Step 4: Future predictions example
def make_prediction(model, features):
    # Example data for prediction
    example_data = pd.DataFrame({
        'energy': [0.7],
        'danceability': [0.8],
        'tempo': [120],
        'valence': [0.6],
        'acousticness': [0.8],
        'liveness': [0.6]
    })

    example_data = example_data[features]

    scaler = StandardScaler()
    example_data_scaled = scaler.fit_transform(example_data)

    recommendation = model.predict(example_data_scaled)
    print(f'\nRecommendation (0: Low, 1: High): {recommendation[0]}')


def main():
    data = load_and_preprocess_data()

    print(data.info())
    print(data.describe())

    visualize_data(data)

    features = ['energy', 'danceability', 'tempo', 'valence', 'acousticness', 'liveness']
    features = [col for col in features if col in data.columns]  # Ensure selected features exist in the data

    model = train_model(data, features)

    make_prediction(model, features)


if __name__ == '__main__':
    main()
