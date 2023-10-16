import numpy as np
import pandas as pd
# 코사인 유사도
def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def fn_sim(user, items):
    sim_arr = []
    for i in items:
        sim_arr.append(round(cos_sim(user, i), 3))
        return sim_arr
    df = pd.read_excel("ITEM_MATRC.xlsx")
    print(df.head())
    print(df.mean())
    user_pro = [1, 0, 0.33, 0.67, 0.67, 0.67, 0.33, 0, 0.67, 0.33]
    item_arr = []
    for i in range(len(df.index)):
        item_arr.append(df.loc[i].tolist())
    print(fn_sim(user_pro, item_arr))