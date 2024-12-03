import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import ast
import cupy as cp  # 引入CuPy库，用于GPU加速

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
# 使用GPU加速计算流行度与评分之间的关系（如果数据量较大）
popularity = cp.array(df['popularity'].values)  # 将数据转移到GPU
vote_average = cp.array(df['vote_average'].values)  # 将数据转移到GPU

# 在GPU上计算评分与流行度的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=popularity.get(), y=vote_average.get())  # 使用.get()将数据从GPU转回到CPU进行绘制
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

# 6. 类型组合与评分的网状图
# 获取所有类型组合（组合每对类型），并计算其平均评分
type_combinations = []

for genres in df['genre_ids']:
    if isinstance(genres, list):
        for comb in combinations(sorted(genres), 2):  # 只考虑2个类型的组合
            type_combinations.append(comb)

# 创建一个 DataFrame 来计算每个类型组合的平均评分
comb_df = pd.DataFrame(type_combinations, columns=['genre_1', 'genre_2'])
comb_df['avg_rating'] = comb_df.apply(lambda row: df[df['genre_ids'].apply(lambda x: row['genre_1'] in x and row['genre_2'] in x)]['vote_average'].mean(), axis=1)

# 创建网状图
G = nx.Graph()

# 为每个类型组合添加边和评分信息
for _, row in comb_df.iterrows():
    G.add_edge(row['genre_1'], row['genre_2'], weight=row['avg_rating'])

# 绘制网状图
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # 使用弹簧布局
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title('Movie Genre Combinations and Average Rating Network')
plt.show()
