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

# 分析评分和流行度按类别（genre_ids）
df_exploded = df.explode('genre_ids')

# 计算每个类别的平均评分和平均流行度
genre_ratings = df_exploded.groupby('genre_ids')['vote_average'].mean()
genre_popularity = df_exploded.groupby('genre_ids')['popularity'].mean()

# 绘制按类别分析的评分和流行度
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_ratings.index, y=genre_ratings.values)
plt.title('Average Rating by Genre')
plt.xlabel('Genre ID')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=genre_popularity.index, y=genre_popularity.values)
plt.title('Average Popularity by Genre')
plt.xlabel('Genre ID')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# 6. 时间趋势分析
# 计算每年发布电影的平均评分和平均流行度
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
yearly_ratings = df.groupby('release_year')['vote_average'].mean()
yearly_popularity = df.groupby('release_year')['popularity'].mean()

# 绘制时间趋势图：评分和流行度
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_ratings)
plt.title('Average Rating Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_popularity)
plt.title('Average Popularity Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.show()

# 7. 类型组合分析与网状图
# 提取类型组合及其平均评分
def get_genre_combinations(genres):
    return list(combinations(genres, 2))

# 转换 'Genres' 列为列表（假设是字符串表示的列表）
df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 提取每部电影的类型组合
genre_combinations = []
for genres in df['Genres']:
    genre_combinations.extend(get_genre_combinations(genres))

# 创建组合 DataFrame
comb_df = pd.DataFrame(genre_combinations, columns=['Genre1', 'Genre2'])
comb_df['average_rating'] = comb_df.apply(
    lambda row: df[df['Genres'].apply(lambda x: row['Genre1'] in x and row['Genre2'] in x)]['vote_average'].mean(),
    axis=1
)

# 去重并重设索引
comb_df = comb_df.drop_duplicates().reset_index(drop=True)

# 创建网状图
G = nx.Graph()

# 添加边及其权重（平均评分）
for _, row in comb_df.iterrows():
    if not pd.isna(row['average_rating']):
        G.add_edge(row['Genre1'], row['Genre2'], weight=row['average_rating'])

# 绘制网状图
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  # 定义布局
nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 添加边标签显示平均评分
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title('Genre Combination Network')
plt.show()
