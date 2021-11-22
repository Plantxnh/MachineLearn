import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
from sklearn.preprocessing import StandardScaler
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
save_path="农药数据\模型调参\50-best_RF.pkl"
#随机森林超参数
n_estimators = 100
criterion = "gini"
max_depth = 10
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.
max_features = "auto"
max_leaf_nodes = None
min_impurity_decrease = 0.
min_impurity_split = None
bootstrap = True
oob_score = False
n_jobs = None
random_state = None
verbose = 0
warm_start = False
class_weight = None
ccp_alpha = 0.0
max_samples = None

name=list(data.columns.values)
Data=np.array(data)
np.random.shuffle(Data)
num_val_sample=len(Data)//k
score_list=[]

#最好的模型
best_forest=RandomForestClassifier()
best_score=0
best_score_list=[]
#K折交叉验证
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    x_train=np.concatenate([Data[:i*num_val_sample,0:feature_num],Data[(i+1)*num_val_sample:,0:feature_num]],axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]], axis=0)
    #随机森林预测
    forest=RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,min_impurity_split=min_impurity_split,bootstrap=bootstrap,oob_score=oob_score,n_jobs=n_jobs,
                 random_state=random_state,verbose=verbose,warm_start=warm_start,class_weight=class_weight,ccp_alpha=ccp_alpha,max_samples=max_samples)
    forest.fit(x_train,y_train.astype('int'))
    score = forest.score(x_test,y_test.astype('int'))
    if best_score<score:
        best_score=score
        best_forest=forest
    score_list.append(score)
print(score_list)
print(sum(score_list)/len(score_list))
#保存最优准确率模型
joblib.dump(best_forest,save_path)
#输出最优准确率
print(best_score)
d2=datetime.datetime.now()
time=d2-d1
print(time)
'''
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    x_train=np.concatenate([Data[:i*num_val_sample,0:feature_num],Data[(i+1)*num_val_sample:,0:feature_num]],axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]], axis=0)
    y_pred = best_forest.predict(x_test)
    score = forest.score(x_test,y_test.astype('int'))
    best_score_list.append(score)
    if best_score<score:
        best_score=score
        y_pred_best=y_pred
print(best_score_list)
print(sum(best_score_list)/len(best_score_list))
'''