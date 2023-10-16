# pip install tensorflow
# 2023-10-16
import tensorflow as tf
import numpy as np
print(tf.__version__)

data = np.loadtxt('../datasets/ThoraricSurgery.csv', delimiter=',')
# 독립변수:환자의 기록 종속변수: 수술후 사망 0, 생존 1
x = data[:, 0:17]#모든행,17개 열
y = data[:, 17] # 모든행, 마지막열
# 딥러닝의 구조 설계
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# 히든레이어 추가
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(5,  activation='relu'))
model.add(Dense(30,  activation='relu'))
# 출력층
model.add(Dense(1, activation='sigmoid')) # 이항분류
model.summary() # 만들 모델의 구조 출력
# 손실함수와 최적화방법 정의
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc']) #metrics 정확도출력
# 학습
model.fit(x, y, epochs=30, batch_size=10)
# 결과 출력
print(f"acc :{model.evaluate(x,y)[1]}")



