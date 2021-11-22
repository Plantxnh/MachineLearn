import numpy as np
import pandas as pd
from sklearn import svm
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
save_path="验证数据\models\soil-smote-best_SVM.pkl"
#SVM超参数
C=1.0
kernel='rbf'
degree=3
gamma='scale'
coef0=0.0
shrinking=True
probability=False
tol=1e-3
cache_size=200
class_weight=None
verbose=False
max_iter=-1
decision_function_shape='ovr'
break_ties=False
random_state=None

name=list(data.columns.values)
Data=np.array(data)
np.random.shuffle(Data)
num_val_sample=len(Data)//k
score_list=[]

#最好的模型
best_SVM=svm.SVC()
best_score=0
best_score_list=[]
#K折交叉验证
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    x_train=np.concatenate([Data[:i*num_val_sample,0:feature_num],Data[(i+1)*num_val_sample:,0:feature_num]],axis=0)
    y_train = np.concatenate([Data[:i * num_val_sample, feature_num:], Data[(i + 1) * num_val_sample:, feature_num:]], axis=0)
    #SVN预测
    SVM=svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,coef0=coef0, shrinking=shrinking, probability=probability,tol=tol,
                cache_size=cache_size, class_weight=class_weight,verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                 break_ties=break_ties,random_state=random_state)
    SVM.fit(x_train,y_train.astype('int'))
    score = SVM.score(x_test,y_test.astype('int'))
    if best_score<score:
        best_score=score
        best_SVM=SVM
    score_list.append(score)
print(score_list)
print(sum(score_list)/len(score_list))
#保存最优准确率模型
joblib.dump(best_SVM,save_path)
#输出最优准确率
print(best_score)
d2=datetime.datetime.now()
time=d2-d1
print(time)