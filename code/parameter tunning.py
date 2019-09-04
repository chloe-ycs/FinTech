import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import xgboost as xgb
import seaborn as sns
pd.set_option('display.max_columns', None)

# import data from pre-prcessing
df_train = pd.read_csv()
y_train = pd.read_csv()
y_test = pd.read_csv()
y_valid = pd.read_csv()

dtrain = pd.read_csv()
dtest = pd.read_csv()
dvalid = pd.read_csv()

#####################################################
### Hyperparameter Tuning - Bayesian Optimization ###
#####################################################
from bayes_opt import BayesianOptimization

train_x = df_train[feat]
train_y = df_train.loanstatus

xgtrain = xgb.DMatrix(train_x, label=train_y, missing = np.NAN)

# step1 - define a target funtion, put the parameters which we want to optimize in it
def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eta'] = 0.05
    params['max_depth'] = int(max_depth )
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = False

    #xgb.cv cross alidation
    cv_result = xgb.cv(params, # hyperparamater
                       xgtrain,
                       num_boost_round=100000,
                       nfold=3,
                       metrics={'auc'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])
    print(cv_result)
    return cv_result['test-auc-mean'].max()

# step2 - bayes optimization object: 第一个参数是我们的优化目标函数，第二个参数是我们所需要输入的超参数名称，以及其范围。超参数名称必须和目标函数的输入名称一一对应
xgb_BO = BayesianOptimization(xgb_evaluate,
                             {'max_depth': (4, 8),
                              'min_child_weight': (0, 20),
                              'colsample_bytree': (0.2, 0.8),
                              'subsample': (0.5, 1),
                              'gamma': (0, 2)
                             }
                            )

# step3 - run bayes optimization, set num of initial data points and iteration times: num of models we will train
xgb_BO.maximize(init_points=5, n_iter=40)

## step4 - Tuning results
xgb_BO.max
xgb_BO.res

xgb_BO_max = pd.DataFrame(xgb_BO.max).T


# parameter set 1
params = {'objective': 'binary:logistic'
                  , 'booster': 'gbtree'
                  , 'eta': 0.01
                  , 'max_depth': 4
                  , 'min_child_weight': 19.530287
                  , 'subsample': 0.50750
                  , 'colsample_bytree': 0.466057
                  , 'gamma': 0.026735
                  , 'seed': 1234
                  , 'nthread': -1
                  , 'silence': 1
                  , 'eval_metric': 'auc'
                  , 'scale_pos_weight': 1}
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
num_boost_round=10000
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=50)


#######################################
#### Retrain model with tuned parameters
######################################
best_xgb_iteration = 1595
clf_train = xgb.XGBClassifier(learning_rate = 0.01
                  , n_estimators = best_xgb_iteration
                  , max_depth = 4
                  , min_child_weight = 19.530287
                  , subsample = 0.50750
                  , colsample_bytree = 0.466057
                  , gamma = 0.026735
                  , seed = 1234
                  , nthread = -1
                  , scale_pos_weight = 1
                  )

clf_train.fit(train_x, train_y)

from self_def_pkg import plot
plot.draw_ROC(gbm, dtrain, dvalid, dtest, y_train, y_valid, y_test)


############################
#### Validate on test data
###########################
y_pred = gbm.predict(dtest)
print (y_pred.max(), y_pred.min(), y_pred.mean())


##########################
### Feature importance ###
##########################
importance2 = gbm.get_fscore()

df_importance2 = pd.DataFrame.from_dict(importance2,orient='index').reset_index()
df_importance2.rename({0:'fscore','index':'feature'},axis=1,inplace=True)
df_importance2['fscore'] = df_importance2['fscore'] / df_importance2['fscore'].sum()
df_importance2.sort_values(['fscore'], ascending=False, inplace=True)
df_importance2

plt.figure(figsize=(32, 32))
df_importance2[:20].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb2.png')



