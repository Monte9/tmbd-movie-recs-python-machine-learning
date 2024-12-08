import pickle
import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬"
)

st.title('Movie Recommender System')
st.write('Find movies similar to your favorites!')

@st.cache_resource
def load_data():
    try:
        movies = pd.DataFrame(pickle.load(open('movie_list.pkl', 'rb')))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies, similarity
    except FileNotFoundError:
        st.error("Required data files not found. Please run create_model.py first.")
        st.stop()

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    recommended_posters = []
    
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]]['id']
        recommended_movies.append(movies.iloc[i[0]]['title'])
        recommended_posters.append(fetch_poster(movie_id))
    
    return recommended_movies, recommended_posters

# Load data
movies, similarity = load_data()

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values,
    index=None,
    placeholder="Choose a movie..."
)

if st.button('Show Recommendations'):
    if selected_movie:
        with st.spinner('Getting recommendations...'):
            recommended_movies, recommended_posters = recommend(selected_movie, movies, similarity)
            
            cols = st.columns(5)
            for col, name, poster in zip(cols, recommended_movies, recommended_posters):
                with col:
                    st.image(poster)
    else:
        st.warning('Please select a movie first!')

st.markdown("Built with ❤️ using Streamlit and TMDB API")