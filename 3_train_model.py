import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import xgboost as xgb
import seaborn as sns
pd.set_option('display.max_columns', None)


######################
### modelling XGBoost
#####################
df = pd.read_csv() # import data from pre-prcessing

feat = list(df.columns.values)
print (feat)

feat.remove('id')
feat.remove('loanstatus')
feat.remove('train_flg')


### Preliminary manually parameter tuning based on stratified train-test split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#Kfolds = StratifiedKFold(df_all['loan_status'], n_folds = 3, shuffle=True, random_state=2019)

df_train = df.query("train_flg == 1")
df_test  = df.query("train_flg == 0" )
print (df_train.shape, df_test.shape)

# 随机划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(df_train[feat], df_train.loanstatus,
                                                      test_size=0.3,
                                                      random_state=2016,
                                                      stratify = df_train.loanstatus ) #stratify：依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
X_test, y_test = df_test[feat], df_test.loanstatus
dtrain = xgb.DMatrix(X_train, y_train, missing = np.NAN)
dvalid = xgb.DMatrix(X_valid, y_valid, missing = np.NAN)
dtest = xgb.DMatrix(X_test, y_test, missing = np.NAN)


params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.05,
          "max_depth": 6,
          "subsample": 0.632,
          "colsample_bytree": 0.7,
          #"colsample_bylevel": 0.6,
          "silent": 1,
          "seed": 1441,
          "eval_metric": "auc",
          #"gamma": 1,
          "min_child_weight": 5} # 74453
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
num_boost_round = 1500


# xgboost.train()利用param列表设置模型参数,原始的xgboost，有cv函数
# 而下面用的xgboost.XGBClassifier()是xgboost的sklearn包。这个包允许我们像GBM一样使用Grid Search和并行处理。
gbm = xgb.train(params,
                dtrain,
                num_boost_round,
                evals = watchlist,
                early_stopping_rounds = 50)


##################
### ROC curve  ###
##################
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model, datasets
import pylab as pl


def draw_ROC(model, dtrain, dvalid, dtest, y_train, y_valid, y_test):
 probas_ = model.predict(dvalid, ntree_limit= model.best_ntree_limit)
 probas_1 = model.predict(dtrain, ntree_limit= model.best_ntree_limit)
 probas_2 = model.predict(dtest, ntree_limit= model.best_ntree_limit)

 fpr, tpr, thresholds = roc_curve(y_valid, probas_) # red
 fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, probas_1)# blue
 fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, probas_2) # green

 roc_auc = auc(fpr, tpr)
 roc_auc_1 = auc(fpr_1, tpr_1)
 roc_auc_2 = auc(fpr_2, tpr_2)

 print("Area under the ROC curve - validation: %f" % roc_auc)
 print("Area under the ROC curve - train: %f" % roc_auc_1)
 print("Area under the ROC curve - test: %f" % roc_auc_2)

 # Plot ROC curve
 plt.figure(figsize=(8, 8))
 plt.plot(fpr, tpr, label='ROC curve - valid(AUC = %0.2f)' % roc_auc, color='r')
 plt.plot(fpr_1, tpr_1, label='ROC curve - train (AUC = %0.2f)' % roc_auc_1, color='b')
 plt.plot(fpr_2, tpr_2, label='ROC curve - test (AUC = %0.2f)' % roc_auc_2, color='g')
 plt.plot([0, 1], [0, 1], 'k--')
 plt.xlim([0.0, 1.0])
 plt.ylim([0.0, 1.0])
 plt.xlabel('False Positive Rate')
 plt.ylabel('True Positive Rate')
 plt.title('ROC for lead score model')
 plt.legend(loc="lower right")
 plt.show()

####
draw_ROC(gbm, dtrain, dvalid, dtest, y_train, y_valid, y_test)

# **accoring to the AUC, curretly a well model. How about tuning parameter by Hypertuning - bayes?



########################
### Predicted values ###
########################
y_pred = gbm.predict(dtest) # gbm: gradient boosting machine
print (y_pred.max(), y_pred.min(), y_pred.mean())
# 0.70249295 0.006733872 0.13376142 就算是你觉得好到板上钉钉的贷款， 它的风险率是0.6%；很不好的贷款，它的风险率是70.2%


### Feature importance
# F-score=(2*precision*recall)/(precision+recall)
importance = gbm.get_fscore() # no feature_importance in xgboost sklearn package，but get_fscore() is the same function
print (importance)
df_importance = pd.DataFrame.from_dict(importance,orient='index').reset_index()
df_importance.rename({0:'fscore','index':'feature'},axis=1,inplace=True)
df_importance['fscore'] = df_importance['fscore'] / df_importance['fscore'].sum()
df_importance.sort_values(['fscore'], ascending=False, inplace=True)
print(df_importance)

# draw top 20 important feature
plt.figure(figsize=(32, 32))
df_importance[:20].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')


grade_importance = df_importance.query("feature=='grade'")
subgrade_importance = df_importance.query("feature=='subgrade'")
intrate_importance = df_importance.query("feature=='intrate'")
df_importance.query("feature=='loanamnt'")


plt.figure(figsize=(32, 32))
df_importance.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')

