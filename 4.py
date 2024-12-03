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

# 5. 类型分析
# 将 genre_ids 转换为 Genres，并计算每个类型的平均评分
df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 创建一个新列，存储每部电影的每个类型
genre_ratings = df.explode('Genres').groupby('Genres')['vote_average'].mean()
genre_popularity = df.explode('Genres').groupby('Genres')['popularity'].mean()

# 绘制类型与评分的关系
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_ratings.index, y=genre_ratings.values)
plt.title('Average Rating by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# 绘制类型与流行度的关系
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_popularity.index, y=genre_popularity.values)
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# 6. 时间趋势分析
# 提取电影发布年份
df['release_year'] = pd.to_datetime(df['release_date']).dt.year

# 按年份分析评分的变化趋势
yearly_ratings = df.groupby('release_year')['vote_average'].mean()

# 绘制年份与评分的关系
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values)
plt.title('Average Rating Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# 7. 网状图分析：根据类型组合绘制电影类型的关系网
# 获取类型组合及其对应的电影平均评分
genre_combinations = df.explode('Genres')[['id', 'Genres']]
genre_combinations = genre_combinations.merge(genre_combinations, on='id', suffixes=('_1', '_2'))

# 只保留不同类型组合（避免自组合）
genre_combinations = genre_combinations[genre_combinations['Genres_1'] < genre_combinations['Genres_2']]

# 计算每个类型组合的平均评分
genre_combinations = genre_combinations.merge(df[['id', 'vote_average']], on='id')
average_ratings = genre_combinations.groupby(['Genres_1', 'Genres_2'])['vote_average'].mean().reset_index()

# 创建网状图
G = nx.Graph()

# 添加边和权重（平均评分）
for _, row in average_ratings.iterrows():
    G.add_edge(row['Genres_1'], row['Genres_2'], weight=row['vote_average'])

# 可视化网状图
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  # 定义布局
nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black')
plt.title('Genre Combination Network (Average Rating)')
plt.show()
