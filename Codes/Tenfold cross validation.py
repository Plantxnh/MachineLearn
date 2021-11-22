import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import joblib
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

#数据路径
path="农药数据\验证集\pesticides-test.csv"
#k折验证
k=10
#特征数量
feature_num=1438
#模型类型“deeplearning or machinelearning
learn="deeplearning"
#数据读入
data=pd.read_csv(path)
#模型路径及名称
save_path="农药数据\models\pesticides-best_model.h5"
Data=pd.read_csv(path)
Data=np.array(Data)
np.random.shuffle(Data)
num_val_sample=len(Data)//k
score_list=[]
best_score=0
#保存图像
AUC_path="农药数据\ROC\pesticides-BPNN-ROC.pdf"
KS_path="农药数据\KS\pesticides-BPNN-KS.pdf"
#导入模型
if learn=="deeplearning":
    model=tf.keras.models.load_model(save_path)
elif learn=="machinelearning":
    model=joblib.load(save_path)
else:
    print("Please ensure the true model type")
    sys.exit()
for i in range(k):
    x_test=Data[i*num_val_sample:(i+1)*num_val_sample,0:feature_num]
    y_test=Data[i*num_val_sample:(i+1)*num_val_sample,feature_num:]
    start=time.time()
    #数据预测
    y_pred=model.predict(x_test)
    end=time.time()
    print("running time:",end-start)
    y_pred=y_pred.astype('int')
    score=accuracy_score(y_test,y_pred)
    if best_score<score:
        best_score=score
        best_model=model
        best_y_test=y_test
        best_y_pred=y_pred
    score_list.append(score)
print("score list:",score_list)
print("best score:",best_score)

fpr, tpr, threshold = metrics.roc_curve(best_y_test, best_y_pred)
roc_auc=metrics.auc(fpr,tpr)
print("ROC:",roc_auc)
KS=tpr-fpr
GINI=2*roc_auc-1
print('GINI:',GINI)
#绘制保存AUC图
plt.figure(1)
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b--', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(AUC_path,dpi=600)
plt.show()
#绘制保存KS图
plt.figure(2)
plt.title('KS')
plt.plot(tpr,'r--',label='tpr')
plt.plot(fpr,'g--',label='fpr')
plt.plot(KS, 'b--', label = 'KS')
plt.legend()
plt.savefig(KS_path,dpi=600)
plt.show()