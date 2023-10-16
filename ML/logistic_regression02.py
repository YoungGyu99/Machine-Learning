import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Titanic 데이터셋을 불러옵니다.
df = pd.read_csv("./datasets/Titanic Passengers.csv")

# 'sex' 열을 여성은 1로, 남성은 0으로 매핑합니다.
df['sex'] = df['sex'].map({'female': 1, 'male': 0})

# 'age' 열의 결측치를 평균 나이로 대체합니다.
df['age'].fillna(value=df['age'].mean(), inplace=True)

# 'pclass' 열을 기반으로 새로운 열을 생성합니다.
df['firstClass'] = df['pclass'].apply(lambda x: 1 if x == 1 else 0)
df['secondClass'] = df['pclass'].apply(lambda x: 1 if x == 2 else 0)
df['thirdClass'] = df['pclass'].apply(lambda x: 1 if x == 3 else 0)

# 입력 변수 (X)와 목표 변수 (y)를 선택합니다.
x = df[['sex', 'age', 'firstClass', 'secondClass', 'thirdClass']]
y = df[['survived']]

# 훈련 데이터와 테스트 데이터로 나눕니다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 데이터를 표준화합니다.
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 로지스틱 회귀 모델을 생성하고 훈련합니다.
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())

# 모델의 계수와 절편을 출력합니다.
print('계수 (Coefficients):', model.coef_)
print('절편 (Intercept):', model.intercept_)

# 모델 성능을 평가합니다.
print('학습 데이터 성능:', model.score(x_train, y_train))
print('테스트 데이터 성능:', model.score(x_test, y_test))

# 샘플 데이터를 예측합니다.
Jack = np.array([0.0, 20.0, 0.0, 0.0, 1.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0, 0.0])
Nick = np.array([0.0, 30.0, 0.0, 1.0, 0.0])
Baby = np.array([1.0, 5.0, 0.0, 1.0, 0.0])
sample = np.array([Jack, Rose, Nick, Baby])
sample = scaler.transform(sample)

# 예측 결과를 출력합니다.
predictions = model.predict(sample)
probabilities = model.predict_proba(sample)

print('예측 결과:', predictions)
print('예측 확률:', probabilities)
