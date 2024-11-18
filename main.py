import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 文件
data = pd.read_csv(r"D:\PycharmProjects\intern\Data\individualMovebankData\2296102400-Branta leucopsis-RRK.csv", header=None, names=["Timestamp", "Latitude", "Longitude"])
# 将时间戳解析为日期时间对象
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
#将timestamp转化为time然后方便模型预测
data["Timestamp"] = data["Timestamp"].apply(lambda x: datetime.timestamp(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')))

x = np.array(data["Timestamp"].values).reshape(-1, 1)
y_lat = np.array(data["Latitude"].values).reshape(-1, 1)
y_long = np.array(data["Longitude"].values).reshape(-1, 1)

#将x, y归一化
scaler = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x)
scaler2 = MinMaxScaler(feature_range=(-1,1))
y_lat = scaler.fit_transform(y_lat)

# 分开测试和训练集
train_size = int(len(x) * 0.8)
test_size = len(x) - train_size
y_train_lat, y_test_lat = y_lat[:train_size], y_lat[train_size:]
x_train_time, x_test_time = x[:train_size], x[train_size:]

# plt.plot(x_train_time, y_train_lat)
# plt.show()


def predict(timestamp):
    # 给定datetime timestamp转化成time然后预测
    datetime_stamp2 = datetime.timestamp(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
    print(datetime_stamp2)

