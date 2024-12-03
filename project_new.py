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

# 按类别分析评分和流行度
genre_ratings = df.explode('genre_ids').groupby('genre_ids')['vote_average'].mean()
genre_popularity = df.explode('genre_ids').groupby('genre_ids')['popularity'].mean()

# 6. 年份分析
# 提取年份信息
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

# 7. 类型组合与评分的网状图
# 提取电影类型组合并计算这些组合的平均评分
def get_genre_combinations(genre_ids):
    # 获取类型的所有可能组合（不考虑顺序）
    return list(combinations(sorted(genre_ids), 2))  # 可以设置组合大小为2

# 获取所有电影的类型组合
genre_combinations = df['genre_ids'].apply(get_genre_combinations)

# 展开所有电影的类型组合
genre_pairs = [pair for sublist in genre_combinations for pair in sublist]

# 创建一个DataFrame来保存每对类型组合的评分
genre_pairs_df = pd.DataFrame(genre_pairs, columns=['genre_1', 'genre_2'])

# 合并评分信息
genre_pairs_df['avg_rating'] = genre_pairs_df.apply(lambda row: df[(df['genre_ids'].apply(lambda x: row['genre_1'] in x and row['genre_2'] in x))]['vote_average'].mean(), axis=1)

# 按类型组合计算平均评分
genre_pairs_avg_rating = genre_pairs_df.groupby(['genre_1', 'genre_2'])['avg_rating'].mean().reset_index()

# 绘制网状图
G = nx.Graph()

# 向图中添加节点和边
genre_df = pd.read_csv('Genre_and_Genre_ID_Mapping.csv')  # Assuming you have a CSV with genre IDs and names
# Create a dictionary to map genre IDs to genre names
genre_dict = pd.Series(genre_df['Genre'].values, index=genre_df['GenreID']).to_dict()
for _, row in genre_pairs_avg_rating.iterrows():
    row['genre_1'] = genre_dict[row['genre_1']]
    row['genre_2'] = genre_dict[row['genre_2']]
    G.add_edge(row['genre_1'], row['genre_2'], weight=row['avg_rating'])

# 绘制图形
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=20)  # 布局
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

# 绘制边
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.6, edge_color=weights, edge_cmap=plt.cm.Blues)

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# 设置标题
plt.title('Network of Genre Combinations and their Average Ratings')
plt.axis('off')
plt.show()
