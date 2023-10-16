
import numpy as np
import matplotlib.pyplot as plt

# x :공부시간, y:성적
x = [2, 4, 6, 8, 10, 12, 14]
y = [1, 1, 1, 1, 0, 0, 0, 0]
x_data = np.array(x)
y_data = np.array(y)
a = 0   #기울기
b = 0   #y절편
lr = 0.05 #학습률
plt.scatter(x_data, y_data)
plt.ylim(-.1, 1.1)
plt.show()
# 시모이드 함수
def sigmoid(x):
    return  1 / (1 + np.e ** (-x))
epochs = 2001 # 학습데이터에 대해서 몇번 학습할지
for i in range(epochs):
    a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
    b_diff = sigmoid(a * x_data + b) - y_data
    a = a - lr * a_diff
    b = b - lr * b_diff
    if i % 100 == 0:
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1),
         np.array([sigmoid(a * x + b ) for x in x_range]))
plt.show()