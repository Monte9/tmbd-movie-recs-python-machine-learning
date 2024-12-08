import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import ast

def main():
    print("Starting movie recommendation model creation...")
    
    # Load datasets
    print("Loading datasets...")
    try:
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
    except FileNotFoundError as e:
        print(f"Error: Dataset files not found: {e}")
        return
    
    # Merge datasets
    print("Merging movie and credits data...")
    movies = movies.merge(credits, on='title')

    # Clean and preprocess the data
    def convert_list(text):
        try:
            return [item['name'] for item in ast.literal_eval(text)]
        except:
            return []

    def get_director(text):
        try:
            crew = ast.literal_eval(text)
            return [person['name'] for person in crew if person['job'] == 'Director'][0]
        except:
            return ''

    def get_cast(text, limit=3):
        try:
            cast = ast.literal_eval(text)
            return [person['name'] for person in cast[:limit]]
        except:
            return []

    print("Processing features...")
    movies['genres'] = movies['genres'].apply(convert_list)
    movies['keywords'] = movies['keywords'].apply(convert_list)
    movies['cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_director)

    # Convert lists to strings
    movies['genres'] = movies['genres'].apply(lambda x: ' '.join(x))
    movies['keywords'] = movies['keywords'].apply(lambda x: ' '.join(x))
    movies['cast'] = movies['cast'].apply(lambda x: ' '.join(x))

    # Create combined feature
    print("Creating combined features...")
    movies['combined_features'] = (
        movies['genres'] + ' ' + 
        movies['keywords'] + ' ' + 
        movies['cast'] + ' ' + 
        movies['director']
    )

    # Create similarity matrix
    print("Calculating similarity matrix...")
    cv = CountVectorizer(stop_words='english', max_features=5000)
    count_matrix = cv.fit_transform(movies['combined_features'].fillna(''))
    similarity = cosine_similarity(count_matrix)

    # Save processed data
    print("Saving processed data...")
    movie_data = movies[['id', 'title']].copy()
    
    try:
        with open('movie_list.pkl', 'wb') as f:
            pickle.dump(movie_data, f)
        print("Successfully saved movie_list.pkl")
        
        with open('similarity.pkl', 'wb') as f:
            pickle.dump(similarity, f)
        print("Successfully saved similarity.pkl")
        
        print("\nModel creation completed!")
        print(f"Total movies processed: {len(movies)}")
    except Exception as e:
        print(f"Error saving files: {e}")
        return

if __name__ == "__main__":
    main()