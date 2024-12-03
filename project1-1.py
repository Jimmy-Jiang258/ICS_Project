import pandas as pd
import seaborn as sns
import ast
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('top_rated_9000_movies_on_TMDB.csv')

# Task 1: Data Cleaning
# Handle missing values
df = df.dropna()

# Handle duplicates
df = df.drop_duplicates()

# Task 2: Movie Rating Analysis
# Analyze average ratings
highest_rated = df[df['vote_average'] == df['vote_average'].max()]
lowest_rated = df[df['vote_average'] == df['vote_average'].min()]

# Task 3: Popularity Analysis
# Relationship between popularity and ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='popularity', y='vote_average')
plt.title('Popularity vs. Average Rating')
plt.xlabel('Popularity')
plt.ylabel('Average Rating')
plt.show()

# Task 4: Genre Analysis
# Analyze ratings and popularity by genre
genre_ratings = df.explode('genre_ids').groupby('genre_ids')['vote_average'].mean()
genre_popularity = df.explode('genre_ids').groupby('genre_ids')['popularity'].mean()

# Task 5: Temporal Analysis
# Trends in ratings and popularity over years
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
yearly_ratings = df.groupby('release_year')['vote_average'].mean()
yearly_popularity = df.groupby('release_year')['popularity'].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_ratings, label='Average Rating')
sns.lineplot(data=yearly_popularity, label='Popularity')
plt.title('Trends in Movie Ratings and Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.show()

# Task 6: Data Visualization
# Distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# Task 7: Recommendation System Basics
# Simple recommendation based on highest ratings
def recommend_movies(n=10):
    return df.nlargest(n, 'vote_average')[['title', 'vote_average']]

# Task 8: Preference-Based Recommendation
# Recommend movies based on user's favorite genres

# 将 'Genres' 列从字符串转换为列表



# Load genre names
genre_df = pd.read_csv('Genre_and_Genre_ID_Mapping.csv')  # Assuming you have a CSV with genre IDs and names
# Create a dictionary to map genre IDs to genre names
genre_dict = pd.Series(genre_df['Genre'].values, index=genre_df['GenreID']).to_dict()
print(genre_dict)
# Map genre IDs to names


favorite_input = input("Enter your favorite movies, genres, or years separated by commas: ").split(',')

def recommend_based_on_favorites(favorite_input, n=10):
    favorite_genres = [genre for genre in favorite_input if genre in genre_dict.values()]
    favorite_years = [int(item) for item in favorite_input if item.isdigit()]
    favorite_titles = [item for item in favorite_input if not item.isdigit() and item not in favorite_genres]
    
    recommended_movies = df[
        df['genre_names'].apply(lambda x: any(genre in x for genre in favorite_genres)) |
        df['release_year'].isin(favorite_years) |
        df['title'].isin(favorite_titles)
    ]
    
    return recommended_movies.nlargest(n, 'vote_average')[['title', 'vote_average']]

# Example usage

# Convert 'genre_ids' to 'genre_names'
df['genre_names'] = df['genre_ids'].apply(lambda x: [genre_dict.get(int(genre_id), 'Unknown') for genre_id in ast.literal_eval(x)])

# Example usage
print(recommend_based_on_favorites(favorite_input, 10))
print(recommend_movies(10))
# New code snippet -----------------------------------------------------------

# 将 'Genres' 列从字符串转换为列表
