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
df['genre_ids'] = df['genre_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 展开每个电影的类型
df_exploded = df.explode('genre_ids')

# 按类型计算平均评分和流行度
genre_ratings = df_exploded.groupby('genre_ids')['vote_average'].mean()
genre_popularity = df_exploded.groupby('genre_ids')['popularity'].mean()

# 可视化各类型的评分
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_ratings.index, y=genre_ratings.values)
plt.title('Average Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# 可视化各类型的流行度
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_popularity.index, y=genre_popularity.values)
plt.title('Average Popularity by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()

# 6. 时间趋势分析
# 按年份计算评分和流行度的平均值
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
yearly_ratings = df.groupby('release_year')['vote_average'].mean()
yearly_popularity = df.groupby('release_year')['popularity'].mean()

# 可视化评分的时间趋势
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values)
plt.title('Average Rating Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# 可视化流行度的时间趋势
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_popularity.index, y=yearly_popularity.values)
plt.title('Average Popularity Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Popularity')
plt.show()

# 6. 类型组合分析与网状图
# 创建一个新的 DataFrame，用于存储每个类型组合的平均评分
type_combinations = []

# 获取所有电影的类型组合
for genres in df['Genres']:
    if len(genres) > 1:
        # 生成类型组合
        for comb in combinations(genres, 2):
            type_combinations.append(sorted(comb))

# 将类型组合列表转换为 DataFrame
type_combinations_df = pd.DataFrame(type_combinations, columns=['Genre_1', 'Genre_2'])

# 计算每对类型组合的平均评分
type_combinations_df['Average_Rating'] = type_combinations_df.apply(
    lambda row: df[(df['Genres'].apply(lambda genres: row['Genre_1'] in genres and row['Genre_2'] in genres))]['vote_average'].mean(),
    axis=1
)

# 设定评分阈值范围
rating_threshold = 7.0  # 可以调整这个值来显示不同评分范围的组合

# 筛选出平均评分大于阈值的类型组合
filtered_combinations = type_combinations_df[type_combinations_df['Average_Rating'] > rating_threshold]

# 构建类型组合的网络图
G = nx.Graph()

# 添加节点和边
for _, row in filtered_combinations.iterrows():
    G.add_edge(row['Genre_1'], row['Genre_2'], weight=row['Average_Rating'])

# 绘制网状图
plt.figure(figsize=(12, 12))

# 设置节点的位置，使用 spring_layout 生成力导向图布局
pos = nx.spring_layout(G, k=0.2, iterations=20)

# 绘制网络
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

# 绘制边上的标签，显示评分值
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f"Genre Combination Network (Avg Rating > {rating_threshold})", fontsize=16)
plt.axis('off')  # 不显示坐标轴
plt.show()

