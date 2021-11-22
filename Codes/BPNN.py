import pandas as pd
import numpy as np

import tensorflow as tf
import datetime
#数据路径
d1=datetime.datetime.now()
path="农药数据\训练集\pesticides-smote-train.csv"
#k折验证
k=10
#特征数量
feature_num=525
#数据读入
data=pd.read_csv(path)
#模型保存路径及名称
save_path="农药数据\模型调参\soil-smote-best_PB.pkl"
#训练次数
epochs=300

name=list(data.columns.values)
Data=np.array(data)
np.random.shuffle(Data)
num_val_sample=len(Data)//k
score_list=[]
#定义模型
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[1,feature_num]))
model.add(tf.keras.layers.Dense(700,activation="relu"))
model.add(tf.keras.layers.Dense(500,activation="relu"))
model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(50,activation="relu"))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(16,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])
#最好的模型
best_BP=model
best_score=0
best_score_list=[]
#K折交叉验证
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    x_train=np.concatenate([Data[:i*num_val_sample,0:feature_num],Data[(i+1)*num_val_sample:,0:feature_num]],axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]], axis=0)
    x_test=tf.convert_to_tensor(x_test,dtype=tf.float32)
    y_test=tf.convert_to_tensor(y_test,dtype=tf.int32)
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    #随机森林预测
    history=model.fit(x_train,y_train,epochs=epochs,validation_data=(x_test,y_test))
    score = history.history['val_accuracy']
    score=np.array(score,dtype=np.float)
    if best_score<score[-1]:
        best_score=score[-1]
        best_BP=model
    score_list.append(score[-1])
print(score_list)
print(sum(score_list)/len(score_list))
#保存最优准确率模型
best_BP.save("农药数据\模型调参\100-best_model.h5")
#输出最优准确率
print(best_score)
d2=datetime.datetime.now()
time=d2-d1
print(time)
