from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

model = load_model('RGTI.model')

a = pd.read_excel('RGTI_20210422_20231019.xlsx', engine='openpyxl')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit_transform(np.array(a['Close'].values).reshape(-1, 1))


df = pd.read_excel('RGTI_20230810_20231020.xlsx', engine='openpyxl')
df['Close'] = scaler.transform(np.array(df['Close'].values).reshape(-1, 1))


test_data = df['Close'].values
pred = model.predict(np.array([test_data]))
print(pred[0][-1])
print(scaler.inverse_transform(np.array([[pred[0][-1]]]).reshape(-1, 1)))
