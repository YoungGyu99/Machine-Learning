import numpy as np
import matplotlib.pyplot as plt

# x: 공부시간, y: 합격/불합격
x = [2, 4, 6, 8, 10, 12, 14]
y = [0, 0, 0, 1, 1, 1, 1]  # 0 불합격, 1 합격
x_data = np.array(x)
y_data = np.array(y)
a = 0  # 기울기
b = 0  # y절편
lr = 0.05  # 학습률

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

epochs = 2001  # 학습데이터에 대해서 몇 번 학습할지
for i in range(epochs):
    for j in range(len(x_data)):
        a_diff = x_data[j] * (sigmoid(a * x_data[j] + b) - y_data[j])
        b_diff = sigmoid(a * x_data[j] + b) - y_data[j]
        a = a - lr * a_diff  # 학습률을 곱하여 a 업데이트
        b = b - lr * b_diff  # 학습률을 곱하여 b 업데이트
    if i % 100 == 0:
        print("epochs=%.f, 기울기 a=%.04f, y절편=%.04f" % (i, a, b))

x_range = np.arange(0, 15, 0.1)
y_pred = sigmoid(a * x_range + b)

plt.scatter(x_data, y_data)
plt.plot(x_range, y_pred, color='red')
plt.xlim(0, 15)
plt.ylim(-0.1, 1.1)
plt.show()
