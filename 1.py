
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
import ast

# 1. 加载数据集
file_path = 'top_rated_9000_movies_on_TMDB.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    raise FileNotFoundError("The dataset 'top_rated_9000_movies_on_TMDB.csv' was not found.")

# 2. 数据清理
# 删除缺失值和重复值
df = df.dropna()
df = df.drop_duplicates()

# 3. 电影评分分析
# 分析最高评分和最低评分的电影
highest_rated = df[df['vote_average'] == df['vote_average'].max()]
lowest_rated = df[df['vote_average'] == df['vote_average'].min()]

# 4. 流行度分析
# 评分与流行度的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='popularity', y='vote_average')
plt.title('Popularity vs. Average Rating')
plt.xlabel('Popularity')
plt.ylabel('Average Rating')
plt.show()

# 5. 类别分析
# 将 'genre_ids' 列从字符串转换为列表
df['genre_ids'] = df['genre_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 6. 计算每种类型的平均评分
# 展开 'genre_ids' 列，按类别计算评分
genre_ratings = df.explode('genre_ids').groupby('genre_ids')['vote_average'].mean().reset_index()

# 可视化：绘制每个类型的平均评分
plt.figure(figsize=(12, 8))
sns.barplot(data=genre_ratings, x='genre_ids', y='vote_average')
plt.title('Average Rating by Genre')
plt.xlabel('Genre IDs')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# 7. 按类别分析评分和流行度
# 按类别计算流行度
genre_popularity = df.explode('genre_ids').groupby('genre_ids')['popularity'].mean().reset_index()

# 合并评分与流行度数据
genre_analysis = pd.merge(genre_ratings, genre_popularity, on='genre_ids')
genre_analysis = genre_analysis.rename(columns={'vote_average': 'avg_rating', 'popularity': 'avg_popularity'})

# 绘制：每个类型的评分与流行度
plt.figure(figsize=(12, 8))
sns.scatterplot(data=genre_analysis, x='avg_popularity', y='avg_rating', hue='genre_ids', palette='Set1', s=100)
plt.title('Average Rating vs Popularity by Genre')
plt.xlabel('Average Popularity')
plt.ylabel('Average Rating')
plt.legend(title='Genre IDs', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 8. 年份分析
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
yearly_ratings = df.groupby('release_year')['vote_average'].mean()
yearly_popularity = df.groupby('release_year')['popularity'].mean()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_ratings, label='Average Rating')
sns.lineplot(data=yearly_popularity, label='Popularity')
plt.title('Trends in Movie Ratings and Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating / Popularity')
plt.legend()
plt.show()
