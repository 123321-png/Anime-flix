import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pre-trained model files
with open('model/similarity.pkl', 'rb') as f:
    similarity_matrix = pickle.load(f)

with open('model/anime_dict.pkl', 'rb') as f:
    anime_dict = pickle.load(f)


# Function to get recommendations based on anime name
def get_recommendations(anime_name, top_n=5):
    # Convert the anime name to the corresponding anime index
    anime_index = None
    for idx, (anime_id, name) in anime_dict.items():
        if name.lower() == anime_name.lower():
            anime_index = idx
            break

    if anime_index is None:
        return []

    # Get the similarity scores for the chosen anime
    similarity_scores = similarity_matrix[anime_index]

    # Sort the scores in descending order and get the top N recommendations
    similar_animes = np.argsort(similarity_scores)[::-1][1:top_n + 1]

    # Get the names of the recommended animes
    recommended_animes = [(anime_dict[anime_id], similarity_scores[anime_id]) for anime_id in similar_animes]

    return recommended_animes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    # Get anime name from the form
    anime_name = request.form['anime_name']

    # Get recommendations
    recommendations = get_recommendations(anime_name)

    return render_template('index.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
