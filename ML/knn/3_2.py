# -*- coding: utf-8 -*-
# k-최근접 이웃 회귀
## 데이터 준비
#114p
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(perch_length
                                  , perch_weight, random_state=1)
train_data = x_train.reshape(-1, 1)
test_data = x_test.reshape(-1, 1)
# 선형회귀와 다항선형회귀
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(train_data, y_train)
# test 50com 농어 무게
print(lr_model.predict([[50]]))
print(lr_model.score(train_data, y_train))
print(lr_model.score(test_data, y_test))

# 2차 방정식으로
train_poly = np.column_stack((train_data**2, train_data))
train_poly = np.column_stack((test_data**2, test_data))

model = LinearRegression()
model.fit(train_poly, y_train)
size = np.arange(15,50)
plt.scatter(train_data, y_train)
plt.xlabel('length')
plt.ylabel('weight')
print(model.score(train_poly, y_train))
print(model.coef_, model.intercept_)
print(model.score(test_poly, y_test))
plt.plot(size, 1.001 * size ** 2 - 23.4 * size + 156.3)
plt.show()

from sklearn.preprocessing import PolynomialFeatures


