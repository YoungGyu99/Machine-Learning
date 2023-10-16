import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
df_ratings = pd.read_csv("ratings.csv")
df_movies = pd.read_csv("movies.csv")
print(df_movies.info())
print(df_ratings.info())
df_ratings.drop('timestamp', axis=1, inplace=True)
print(df_ratings.head())
user_item_rating = pd.merge(df_ratings, df_movies, on="movieId")
print(user_item_rating.head())
movie_matrix = user_item_rating.pivot_table("rating"
                                            , index="title",columns="userId")
print(movie_matrix.head())
# 널값 0으로
movie_matrix.fillna(0, inplace=True)
# 유사도 비교
item_cf = cosine_similarity(movie_matrix)
print(item_cf)
result_df = pd.DataFrame(data=item_cf, index=movie_matrix.index
                         , columns=movie_matrix.index)
print(result_df)
def get_item_based(title):
    # asc 오름 False-> desc
    return  result_df[title].sort_values(ascending=False)[:10]
while True:
    text = input("좋아하는 영화 이름을 정확하게 입력하세요:")
    print(get_item_based(text))