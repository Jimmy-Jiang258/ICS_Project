from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import ast
from difflib import get_close_matches

app = Flask(__name__)
CORS(app)  # 允许所有来源的跨域请求

# 加载数据集
df = pd.read_csv('top_rated_9000_movies_on_TMDB.csv')
genre_df = pd.read_csv('Genre_and_Genre_ID_Mapping.csv')
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
genre_dict = pd.Series(genre_df['Genre'].values, index=genre_df['GenreID']).to_dict()
df['genre_names'] = df['genre_ids'].apply(lambda x: [genre_dict.get(int(genre_id), 'Unknown') for genre_id in ast.literal_eval(x)])

def recommend_based_on_favorites(favorite_input, n=30):
    favorite_genres = [genre for genre in favorite_input if genre in genre_dict.values()]
    favorite_years = [int(item) for item in favorite_input if item.isdigit()]
    favorite_titles = [item for item in favorite_input if not item.isdigit() and item not in favorite_genres]
    
    if favorite_titles:
        favorite_movies = df[df['title'].isin(favorite_titles)]
        if favorite_movies.empty:
            # Find close matches for the given titles
            all_titles = df['title'].tolist()
            close_matches = []
            for title in favorite_titles:
                matches = get_close_matches(title, all_titles, n=3, cutoff=0.6)
                close_matches.extend(matches)
            favorite_movies = df[df['title'].isin(close_matches)]
        
        if not favorite_movies.empty:
            favorite_genres = favorite_movies.explode('genre_names')['genre_names'].unique()
        else:
            print("No matching movies found for the given titles.")
            return []
    
    recommended_movies = df[
        df['genre_names'].apply(lambda x: any(genre in x for genre in favorite_genres)) |
        df['release_year'].isin(favorite_years) |
        (df['genre_names'].apply(lambda x: any(genre in x for genre in favorite_genres)) & 
        ~df['title'].isin(favorite_titles))
    ]
    
    return recommended_movies.nlargest(n, 'vote_average')[['title', 'vote_average', 'overview']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    favorite_input = data['favorites'].split(',')
    recommendations = recommend_based_on_favorites(favorite_input)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)