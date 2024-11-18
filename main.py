import pandas as pd
import numpy as np
from datetime import datetime

# 读取 CSV 文件
data = pd.read_csv(r"D:\PycharmProjects\intern\Program\predict\2296102400-Branta leucopsis-RRK.csv", header=None, names=["Timestamp", "Latitude", "Longitude"])
# 将时间戳解析为日期时间对象
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
#将timestamp转化为time然后方便模型预测
data["Timestamp"] = data["Timestamp"].apply(lambda x: datetime.timestamp(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')))
# 输出特征
locations = data[["Latitude", "Longitude"]].values



# 分开测试和训练集
train_size = int(len(locations) * 0.67)
test_size = len(locations) - train_size
train, test = locations[:train_size], locations[train_size:]
print(train)

def predict(timestamp):
    # 给定datetime timestamp转化成time然后预测
    datetime_stamp2 = datetime.timestamp(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
    print(datetime_stamp2)

