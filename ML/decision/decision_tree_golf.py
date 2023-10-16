# 의사결정나무 시각화를 위한 라이브러리 설치
# pip install graphviz
import  pandas as pd
from sklearn import  tree
from sklearn.preprocessing import  LabelEncoder

df = pd.read_csv("playing golf.csv")
print(df.columns)
x_data = df[['outlook', 'temperature','humidity','windy']]
y_data = df[['play']]
enc_class ={}
def encoding_label(x):
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_class[x.name] = le.classes_
    return  label
train_data = x_data[x_data.columns].apply(encoding_label)
print(train_data)

model = tree.DecisionTreeClassifier()
model.fit(train_data, y_data)
test = pd.DataFrame({"outlook":2, "temperature":1, "humidity":0, "windy":0, "windy":0}, index=[0])
pred = model.predict(test)
prob = model.predict_proba(test)
print(pred, prob)
# 트리시각화
import graphviz
print(train_data.columns.values.tolist())
print(y_data.drop_duplicates().values.tolist())
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plot_tree(model, feature_names=['outlook','temperature','humidity','windy'])
print(enc_class)
plt.show()

# dot_data = tree.export_graphviz(model, out_file=None
#                                  ,feature_names=train_data.columns.values.tolist()
#                                 ,class_names=y_data.drop_duplicates().values.tolist()
#                                 ,filled=True, rounded=True,special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render('./golf', view=True)
# print(graph)
