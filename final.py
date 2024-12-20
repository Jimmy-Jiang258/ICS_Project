import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import combinations
import ast
from difflib import get_close_matches

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
# 绘制类型评分的箱型图
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.explode('Genres'), x='Genres', y='vote_average')
plt.title('Boxplot of Ratings by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# -----------------------------------------------------------------------------------
# plt.figure(figsize=(12, 6))
# sns.boxplot(x=genre_ratings.index, y=genre_ratings.values)
# plt.title('Average Rating by Genre')
# plt.xlabel('Genre')
# plt.ylabel('Average Rating')
# plt.xticks(rotation=90)
# plt.show()
# -----------------------------------------------------------------------------------
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
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
yearly_ratings = df.groupby('release_year')['vote_average'].mean()
yearly_popularity = df.groupby('release_year')['popularity'].mean()

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=yearly_popularity, label='Popularity')
plt.title('Trends in Movie Ratings and Popularity Over Years')
plt.xlabel('Year')
plt.ylabel('Popularity')
ax2 = ax1.twinx()
sns.lineplot(data=yearly_ratings, label='Average Rating', color='red')
ax2.set_ylabel('Rating')
ax2.tick_params(axis='y')
ax2.set_ylim(ax2.get_ylim()[0], ax2.get_ylim()[1])
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['vote_average'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()

# 7. 类型组合分析与网状图
# -----------------------------------------------------------------------------------

# # 提取类型组合及其平均评分
# def get_genre_combinations(genres):
#     return list(combinations(genres, 2))

# # 转换 'Genres' 列为列表（假设是字符串表示的列表）
# df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# # 提取每部电影的类型组合
# genre_combinations = []
# for genres in df['Genres']:
#     genre_combinations.extend(get_genre_combinations(genres))

# # 创建组合 DataFrame
# comb_df = pd.DataFrame(genre_combinations, columns=['Genre1', 'Genre2'])
# comb_df['average_rating'] = comb_df.apply(
#     lambda row: df[df['Genres'].apply(lambda x: row['Genre1'] in x and row['Genre2'] in x)]['vote_average'].mean(),
#     axis=1
# )

# # 去重并重设索引
# comb_df = comb_df.drop_duplicates().reset_index(drop=True)

#新代码------------------------------------------------------------------------------------------------------------------------------
# type_combinations = []

# # 获取所有电影的类型组合
# for genres in df['Genres']:
#     if len(genres) > 1:
#         # 生成类型组合
#         for comb in combinations(genres, 2):
#             type_combinations.append(sorted(comb))

# # 将类型组合列表转换为 DataFrame
# type_combinations_df = pd.DataFrame(type_combinations, columns=['Genre_1', 'Genre_2'])

# # 计算每对类型组合的平均评分
# type_combinations_df['Average_Rating'] = type_combinations_df.apply(
#     lambda row: df[(df['Genres'].apply(lambda genres: row['Genre_1'] in genres and row['Genre_2'] in genres))]['vote_average'].mean(),
#     axis=1
# )

# # 设定评分阈值范围
# rating_threshold = 7.0  # 可以调整这个值来显示不同评分范围的组合

# # 筛选出平均评分大于阈值的类型组合
# filtered_combinations = type_combinations_df[type_combinations_df['Average_Rating'] > rating_threshold]

# # -----------------------------------------------------------------------------------


# 创建网状图
# G = nx.Graph()

# # 添加边及其权重（平均评分）
# for _, row in comb_df.iterrows():
#     if not pd.isna(row['average_rating']):
#         G.add_edge(row['Genre1'], row['Genre2'], weight=row['average_rating'])

# # 绘制网状图
# plt.figure(figsize=(12, 10))
# pos = nx.spring_layout(G, seed=42)  # 定义布局
# nx.draw_networkx_nodes(G, pos, node_size=5000, node_color='skyblue')
# nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# # 添加边标签显示平均评分
# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# # plt.title(f"Genre Combination Network (Avg Rating > {rating_threshold})", fontsize=16)
# plt.title('Genre Combination Network')
# plt.show()

# -----------------------------------------------------------------------------------
# def get_genre_combinations(genre_ids):
#     # 获取类型的所有可能组合（不考虑顺序）
#     return list(combinations(sorted(genre_ids), 2))  # 可以设置组合大小为2

# # 获取所有电影的类型组合
# genre_combinations = df['genre_ids'].apply(get_genre_combinations)

# # 展开所有电影的类型组合
# genre_pairs = [pair for sublist in genre_combinations for pair in sublist]

# # 创建一个DataFrame来保存每对类型组合的评分
# genre_pairs_df = pd.DataFrame(genre_pairs, columns=['genre_1', 'genre_2'])

# # 合并评分信息
# genre_pairs_df['avg_rating'] = genre_pairs_df.apply(lambda row: df[(df['genre_ids'].apply(lambda x: row['genre_1'] in x and row['genre_2'] in x))]['vote_average'].mean(), axis=1)

# # 按类型组合计算平均评分
# genre_pairs_avg_rating = genre_pairs_df.groupby(['genre_1', 'genre_2'])['avg_rating'].mean().reset_index()

# # 绘制网状图
# G = nx.Graph()

# # 向图中添加节点和边
# genre_df = pd.read_csv('Genre_and_Genre_ID_Mapping.csv')  # Assuming you have a CSV with genre IDs and names
# # Create a dictionary to map genre IDs to genre names
# genre_dict = pd.Series(genre_df['Genre'].values, index=genre_df['GenreID']).to_dict()
# for _, row in genre_pairs_avg_rating.iterrows():
#     row['genre_1'] = genre_dict[row['genre_1']]
#     row['genre_2'] = genre_dict[row['genre_2']]
#     G.add_edge(row['genre_1'], row['genre_2'], weight=row['avg_rating'])

# # 绘制图形
# plt.figure(figsize=(12, 8))
# pos = nx.spring_layout(G, k=0.5, iterations=20)  # 布局
# edges = G.edges()
# weights = [G[u][v]['weight'] for u, v in edges]

# # 绘制边
# nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.6, edge_color=weights, edge_cmap=plt.cm.Blues)

# # 绘制节点
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)

# # 绘制标签
# nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# # 设置标题
# plt.title('Network of Genre Combinations and their Average Ratings')
# plt.axis('off')
# plt.show()

# -----------------------------------------------------------------------------------   

def recommend_movies(n=10):
    return df.nlargest(n, 'vote_average')[['title', 'vote_average']]

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
            return pd.DataFrame()  # Return an empty DataFrame if no matches are found
    
    recommended_movies = df[
        df['genre_names'].apply(lambda x: any(genre in x for genre in favorite_genres)) |
        df['release_year'].isin(favorite_years) |
        (df['genre_names'].apply(lambda x: any(genre in x for genre in favorite_genres)) & 
        ~df['title'].isin(favorite_titles))
    ]
    
    return recommended_movies.nlargest(n, 'vote_average')[['title', 'vote_average']]

df['genre_names'] = df['genre_ids'].apply(lambda x: [genre_dict.get(int(genre_id), 'Unknown') for genre_id in ast.literal_eval(x)])

# Example usage
recommended_movies = recommend_based_on_favorites(favorite_input, 10)
if not recommended_movies.empty:
    print(recommended_movies)

print(recommend_movies(10))