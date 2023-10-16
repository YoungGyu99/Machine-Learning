import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
df_ratings = pd.read_csv("ratings.csv")
df_movies = pd.read_csv("movies.csv")
df_ratings.drop('timestamp', axis=1, inplace=True)
user_item_rating = pd.merge(df_ratings, df_movies, on="movieId")
user_matrix = user_item_rating.pivot_table("rating"
                                            , index="userId",columns="title")
print(user_matrix.head())
user_matrix.fillna(0, inplace=True)
user_cf = cosine_similarity(user_matrix)
result_df = pd.DataFrame(data=user_cf
                         , index=user_matrix.index
                         , columns=user_matrix.index)
print(result_df.shape)

# 대상 유저와 유사한 유저의 평점 높은 영화(본 영화 제외)
def get_user(id, userId):
    movie_arr = user_item_rating[user_item_rating['userId'] == id]
    user_watch_movie = user_item_rating[user_item_rating['userId'] == userId]
    # ~ 제외의 의미 <- 안하면 포함
    movie_arr = movie_arr[~movie_arr['movieId'].isin(
        user_watch_movie['movieId'].values.tolist())]
    five_movie = movie_arr.sort_values(by='rating', ascending=False)[:6]
    return five_movie['title'].values.tolist()

print(get_user(1, 10))
# 추천 대상의 id 해당 아이디와 유사한 유저의 평점 높은 영화 중 내가 안본 영화 리스트
def get_user_itme(id):
    best = user_item_rating[user_item_rating['userId']== id].sort_values(by='rating'
                                                                             ,ascending=False)[:5]
    print("my best")
    print(best['title'])
    sim_user = result_df[id].sort_values(ascending=False)[:6]
    sim_id_arr = sim_user.index.tolist()[1:]
    data = []
    for i in sim_id_arr:
        print("sim user:" + str(i))
        item = get_user(i, id)
        data = data + item
    return set(data) #유사한 유저들의 평점높은 영화중 중복이 있으면 제거
while True:
    text = input("추천 대상 유저의 id")
    print(get_user_itme(int(text)))