import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import  MinMaxScaler
import datetime

# 데이터 셋 만들기 x: 50일 y: 51
df = pd.read_excel('RGTI_20210422_20231019.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
# 툭성의 값을 특정범위 ex 0 ~ 1 사이로 변환하는데 사용
df['Close'] = scaler.fit_transform(np.array(df['Close'].values.reshape(-1, 1)))


data_cnt = len(df['Close'].values)
seq_len = 50
seq_all = seq_len + 1
result = []
for idx in range(data_cnt - seq_all):
    result.append(df['Close'].values[idx : idx + seq_all])
print(result)
result = np.array(result)
row_cnt = int(round(result.shape[0] * 0.9))  # 10% 훈련 데이터로 사용 건수
train_data = result[:row_cnt, :]

x_train = train_data[:, :-1]
x_train_reshape = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train_data[:, -1]
print(x_train_reshape)

test_data = result[row_cnt:, :-1]
x_test_reshape = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
y_test = result[row_cnt:, -1]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train_reshape, y_train, validation_data=(x_test_reshape, y_test)
          , batch_size=10
          , epochs=20)
model.save('RGTI.model')
pred = model.predict(x_test_reshape)
import matplotlib.pyplot as plt
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
