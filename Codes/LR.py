import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
import joblib
import datetime


#数据路径
d1=datetime.datetime.now()
path="验证数据\训练集\soil-smote-train.csv"
#k折验证
k=10
#特征数量
feature_num=263
#数据读入
data=pd.read_csv(path)
#模型保存路径及名称
save_path="验证数据\models\soil-smote-best_LR.pkl"
#逻辑回归超参数
penalty='l2'
dual=False
tol=1e-4
C=1.0
fit_intercept=True
intercept_scaling=1
class_weight=None
random_state=None
solver='lbfgs'
max_iter=100
multi_class='auto'
verbose=0
warm_start=False
n_jobs=None
l1_ratio=None

name=list(data.columns.values)
Data=np.array(data)
np.random.shuffle(Data)
num_val_sample=len(Data)//k
score_list=[]

#最好的模型
best_LR=LogisticRegression()
best_score=0
best_score_list=[]
#K折交叉验证
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    x_train=np.concatenate([Data[:i*num_val_sample,0:feature_num],Data[(i+1)*num_val_sample:,0:feature_num]],axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]], axis=0)
    #逻辑回归预测
    LR=LogisticRegression(penalty=penalty,dual=dual, tol=tol, C=C,fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                 l1_ratio=l1_ratio)
    LR.fit(x_train,y_train.astype('int'))
    score = LR.score(x_test,y_test.astype('int'))
    if best_score<score:
        best_score=score
        best_LR=LR
    score_list.append(score)
print(score_list)
print(sum(score_list)/len(score_list))
#保存最优准确率模型
joblib.dump(best_LR,save_path)
#输出最优准确率
print(best_score)
d2=datetime.datetime.now()
time=d2-d1
print(time)