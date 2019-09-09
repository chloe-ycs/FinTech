

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import xgboost as xgb
import seaborn as sns
pd.set_option('display.max_columns', None)

#################
### Load Data ###
#################
df1 = pd.read_csv('/Users/chloe.song/Documents/Projects/11FinTech/loan_2014.csv')
'''
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df1 = df1.select_dtypes(include=numerics)
num_feature = list(num_df1)

obj = ['object']
ob_df1 = df1.select_dtypes(include=obj)
ob_feature = list(ob_df1)
'''

all_null_feature =['inqlast12m',
 'verificationstatusjoint',
 'maxbalbc',
 'annualincjoint',
 'memberid',
 'openil6m',
 'openacc6m',
 'totalbalil',
 'inqfi',
 'mthssincercntil',
 'openrv24m',
 'openrv12m',
 'dtijoint',
 'openil12m',
 'openil24m',
 'allutil',
 'totalcutl',
 'ilutil']
ob_feature=['id',
 'zipcode',
 'revolutil',
 'intrate',
 'applicationtype',
 'grade',
 'term',
 'subgrade',
 'earliestcrline',
 'initialliststatus',
 'purpose',
 'emptitle',
 'verificationstatus',
 'addrstate',
 'homeownership',
 'emplength',
 'issued',
 'loanstatus']
num_feature=['avgcurbal',
 'numrevaccts',
 'mortacc',
 'numactvrevtl',
 'totalilhighcreditlimit',
 'numiltl',
 'mthssincelastdelinq',
 'totalacc',
 'revolbal',
 'numacctsever120pd',
 'pcttlnvrdlq',
 'numtl30dpd',
 'percentbcgt75',
 'numoprevtl',
 'bcutil',
 'numactvbctl',
 'bcopentobuy',
 'totalrevhilim',
 'numrevtlbalgt0',
 'mthssincerecentrevoldelinq',
 'tothicredlim',
 'fundedamnt',
 'numtl90gdpd24m',
 'mthssincerecentinq',
 'numbcsats',
 'installment',
 'totalbclimit',
 'mosinoldrevtlop',
 'numsats',
 'inqlast6mths',
 'mthssincerecentbc',
 'mosinrcnttl',
 'loanamnt',
 'mthssincerecentbcdlq',
 'totalbalexmort',
 'numbctl',
 'openacc',
 'dti',
 'totcollamt',
 'numtloppast12m',
 'annualinc',
 'totcurbal',
 'accnowdelinq',
 'pubrecbankruptcies',
 'chargeoffwithin12mths',
 'collections12mthsexmed',
 'numtl120dpd2m',
 'accopenpast24mths',
 'mthssincelastrecord',
 'taxliens',
 'mosinoldilacct',
 'delinq2yrs',
 'mosinrcntrevtlop',
 'pubrec',
 'delinqamnt',
 'mthssincelastmajorderog']

select_feature = ob_feature + num_feature

# remove applicationtype since it only has one value as 'individual'
ob_feature.remove('applicationtype')

# remove fundedamnt since it always equals to 1 for issued loans and varied during different time of a current loan
num_feature.remove('fundedamnt')

# remove 'id','issued' and 'loanstatus' from ob_feature, since they are index, train/test flag and target
ob_feature.remove('id')
ob_feature.remove('loanstatus')
ob_feature.remove('issued')

select_feature = ob_feature + num_feature

# selected features left and delete the last two rows
df = df1[ select_feature +['id','loanstatus','issued']][:-2]


##################################
##### only use term=36 months and loanstatus = fully paid and charged off
###################################
df = df.query("loanstatus == 'Fully Paid' or loanstatus == 'Charged Off' ")
df = df.query("term ==' 36 months'")

df.drop('term', axis = 1, inplace=True)
ob_feature.remove('term')


###################
### Label Target ##
###################
df['loanstatus'] = df.loanstatus.map({"Charged Off": 1, "Fully Paid": 0})
print (df.loanstatus.value_counts())
print (df.loanstatus.value_counts(normalize=True))

### Out of time testing set and in-time training set: Oct~Dec as test)
issued = list(df['issued'].unique())

df['train_flg'] = df.issued.apply(lambda x: 0 if x in issued[:3] else 1) #train_flg = 1：train set; 0:test set

df.drop('issued', axis = 1, inplace=True)



#####################################
### Feature Engineering & Cleaning
#####################################
#### 1- Datetime to numeric feature
df.earliestcrline.unique()[:5] # earliestcrlin： 最早有信用的时间

# convert to number of months to Dec 2014
cl_month = df.earliestcrline.apply(lambda x: x.split('-')[0])
cl_year = df.earliestcrline.apply(lambda x: int(x.split('-')[1]))

# 到2014年12月为止，具有信用历史几个月了 2014-1989*12+(12-9)
dic_month= {'Jan':11,'Feb':10,'Mar':9,'Apr':8, 'May':7, 'Jun':6, 'Jul':5, 'Aug':4, 'Sep':3, 'Oct':2, 'Nov':1, 'Dec':0}
df['earliestcrline_month'] = df.earliestcrline.apply(lambda x: (2014 - int(x.split('-')[1]))*12 + dic_month[x.split('-')[0]] )
#df.earliestcrline[:5]
#df.earliestcrline_month[:5]

num_feature.append('earliestcrline_month')
ob_feature.remove('earliestcrline')

print (len(ob_feature), len(num_feature)) # 12 56

df.drop('earliestcrline', axis = 1, inplace=True)

#### 2- emplength to numeric feature
print (df['emplength'].isnull().sum())
df['emplength'].replace('n/a', np.nan, inplace=True) # na: no content，default no job
df['emplength'].replace('< 1 year', '0', inplace=True)
df['emplength'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df.emplength.fillna(value=-999, inplace=True) # -999想表示空值是一个不好的事情，留空在xgboost也可以
df['emplength'] = df['emplength'].astype(int)
#df['emplength'].unique()
#df.emplength.value_counts()

num_feature.append('emplength')
ob_feature.remove('emplength')
print (len(num_feature), len(ob_feature)) # 57 11

### 3- % to numeric feature: intrate利率, revolutil
intrate = df.intrate.apply(lambda x: float(x.replace('%','')))
df.intrate = intrate
# df.intrate[:2]
df.revolutil = pd.Series(df.revolutil).str.replace('%', '').astype(float)
#df.revolutil[:2]

ob_feature.remove('intrate')
num_feature.append('intrate')
ob_feature.remove('revolutil')
num_feature.append('revolutil')
print (len(ob_feature), len(num_feature)) # 9 59


#### 4- Ordinal feature encoding : grade, subgrade
Dic_grade = {"A": 1, "B": 2,"C": 3, "D": 4, "E": 5,"F": 6, "G": 7}
df.grade = df.grade.map(Dic_grade)
df.subgrade = df.subgrade.apply(lambda x: (Dic_grade[x[0]] - 1) * 5 + int(x[1])) # C3: (3-1)*5+3=13

ob_feature.remove('grade')
num_feature.append('grade')
ob_feature.remove('subgrade')
num_feature.append('subgrade')
print (len(ob_feature), len(num_feature)) # 7 61


### High cardinality feature encoding
#### 5- Zip Code - frequency encoding
print (df.zipcode.nunique(), df.zipcode.unique()[:5])
df.zipcode = df.zipcode.apply(lambda x: int(x[0:3]))

# 每一个zipcode出现的次数
zipcode_freq = df.groupby("zipcode").size().reset_index()
#zipcode_freq[:5]

zipcode_freq.columns = ["zipcode", "zipcode_freq"]

df = pd.merge(df, zipcode_freq, how = "left", on = "zipcode")

ob_feature.remove('zipcode')
num_feature.append('zipcode_freq')
num_feature.append('zipcode')
print (len(ob_feature), len(num_feature)) # 6 63


#### 6- emptitle-frequency encoding (can do some NLP for later stage)
emptitle_freq = df.groupby("emptitle").size().reset_index()
emptitle_freq.columns = ["emptitle", "emptitle_freq"]
df = pd.merge(df, emptitle_freq, how = "left", on = "emptitle")

df.drop("emptitle", axis = 1, inplace=True)

ob_feature.remove('emptitle')
num_feature.append('emptitle_freq')
print (len(ob_feature), len(num_feature)) # 5 64


#### 7- Addr_state - frequency encoding
addrstate_freq = df.groupby("addrstate").size().reset_index()
addrstate_freq.columns = ["addrstate", "addrstate_freq"]
df = pd.merge(df, addrstate_freq, how = "left", on = "addrstate")

df.drop("addrstate", axis = 1, inplace=True)

ob_feature.remove('addrstate')
num_feature.append('addrestate_freq')
print (len(ob_feature), len(num_feature)) # 4 65


### 8- One hot encoding
dummy_feature = ["homeownership", "verificationstatus", "purpose", "initialliststatus"]
df_dummy = pd.get_dummies(df[dummy_feature])
df = pd.concat([df,df_dummy], axis=1 )
df.drop(dummy_feature, axis = 1, inplace=True)



######################
### modelling XGBoost
##################

feat = list(df.columns.values)
print (feat)

feat.remove('id')
feat.remove('loanstatus')
feat.remove('train_flg')

###Preliminary manually parameter tuning based on stratified train-test split

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#Kfolds = StratifiedKFold(df_all['loan_status'], n_folds = 3, shuffle=True, random_state=2019)

df_train = df.query("train_flg == 1")
df_test  = df.query("train_flg == 0" )
print (df_train.shape, df_test.shape) #(112550, 90) (50020, 90)

# 随机划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(df_train[feat], df_train.loanstatus,
                                                      test_size=0.3,
                                                      random_state=2016,
                                                      stratify = df_train.loanstatus ) #stratify：依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
#X_train.info()

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
# Best iteration:train-auc:0.781018	eval-auc:0.69215
# gbm.best_iteration = 1595; gbm.best_ntree_limit=1596


'''
xgb.train()参数
https://www.cnblogs.com/Allen-rg/p/10563362.html

params 这是一个字典，里面包含着训练中的参数关键字和对应的值，形式是params = {‘booster’:’gbtree’,’eta’:0.1}
dtrain 训练的数据
obj,自定义目的函数
feval,自定义评估函数
num_boost_round 这是指提升迭代的个数
evals 这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals = [(dtrain,’train’),(dval,’val’)]
或者是evals = [(dtrain,’train’)],对于第一种情况,它使得我们可以在训练过程中观察验证集的效果。

evals_result 字典，存储在watchlist中的元素的评估结果。

early_stopping_rounds,当设置的迭代次数较大时，可在一定的迭代次数内准确率没有提升就停止训练
要求evals里至少有一个元素,如果有多个,按最后一个去执行,返回的是最后的迭代次数（不是最好的）。
如果early_stopping_rounds存在,则模型会生成三个属性: 
bst.best_score, bst.best_iteration, bst.best_ntree_limit

verbose_eval (可以输入布尔型或数值型)，也要求evals里至少有一个元素。
如果为True,则对evals中元素的评估结果会输出在结果中；如果输入数字，假设为5，则每隔5个迭代输出一次。

xgb_model ,在训练之前用于加载的xgb model。

'''

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

 fpr, tpr, thresholds = roc_curve(y_valid, probas_) # 红色
 fpr_1, tpr_1, thresholds_1 = roc_curve(y_train, probas_1)# 蓝色
 fpr_2, tpr_2, thresholds_2 = roc_curve(y_test, probas_2) # 绿色

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

# 不调参的话交叉验证AUC均值如图，算是一个不错的模型，那么如果用bayes调参结果会怎么样呢？↓



########################
### Predicted values ###
########################
y_pred = gbm.predict(dtest)
print (y_pred.max(), y_pred.min(), y_pred.mean())


### Feature importance
# F-score=(2*precision*recall)/(precision+recall)
importance = gbm.get_fscore() # xgboost的sklearn包没有feature_importance这个量度，但是get_fscore()函数有相同的功能
print (importance)
df_importance = pd.DataFrame.from_dict(importance,orient='index').reset_index()
df_importance.rename({0:'fscore','index':'feature'},axis=1,inplace=True)
df_importance['fscore'] = df_importance['fscore'] / df_importance['fscore'].sum()
#df_importance
df_importance.sort_values(['fscore'], ascending=False, inplace=True)
#df_importance

# 把重要性前10画出来
plt.figure(figsize=(32, 32))

df_importance[:10].sort_values(by='fscore').plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
plt.tick_params(labelsize = 12)
plt.xlabel('Relative Importance', fontsize = 12)
plt.ylabel('Feature Name', fontsize = 12)
plt.title('XGBoost Feature Importance',fontsize = 16)
plt.gcf().savefig('feature_importance_xgb.png')


grade_importance = df_importance.query("feature=='grade'")
subgrade_importance = df_importance.query("feature=='subgrade'")
intrate_importance = df_importance.query("feature=='intrate'")
df_importance.query("feature=='loanamnt'")

plt.figure(figsize=(32, 32))
df_importance.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')


#####################################################
### Hyperparameter Tuning - Bayesian Optimization ###
#####################################################
from bayes_opt import BayesianOptimization

train_x = df_train[feat]
train_y = df_train.loanstatus

xgtrain = xgb.DMatrix(train_x, label=train_y, missing = np.NAN)

# step1- 定义一个目标函数，里面放入我们希望优化的函数
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

    #xgb.cv交叉验证
    cv_result = xgb.cv(params, # 超参数
                       xgtrain,
                       num_boost_round=100000,
                       nfold=3,
                       metrics={'auc'},
                       seed=1234,
                       callbacks=[xgb.callback.early_stop(50)])
    print(cv_result)
    # 为了实现提前停止的交叉验证，我们使用xgboost函数cv，它输入为超参数，训练集，用于交叉验证的折数等
    # 我们将迭代次数（num_boost_round）设置为10000，但实际上不会达到这个数字，因为我们使用callbacks来停止训练，当连续50轮迭代效果都没有提升时，则提前停止，并选择模型。
    # 因此，迭代次数并不是我们需要设置的超参数。

    # 一旦交叉验证完成，我们就会得到最好的分数roc-auc,然后返回这个值↓
    return cv_result['test-auc-mean'].max()

# step2- bayes优化对象: 第一个参数是我们的优化目标函数，第二个参数是我们所需要输入的超参数名称，以及其范围。超参数名称必须和目标函数的输入名称一一对应
xgb_BO = BayesianOptimization(xgb_evaluate,
                             {'max_depth': (4, 8),
                              'min_child_weight': (0, 20),
                              'colsample_bytree': (0.2, 0.8),
                              'subsample': (0.5, 1),
                              'gamma': (0, 2)
                             }
                            )

# step3- 运行bayes优化, 设置初始点的数量，以及我们想要的迭代次数，迭代次数将是我们要训练的机器学习模型数量
xgb_BO.maximize(init_points=5, n_iter=40)  # bayes_opt库只支持最大值

## step4- Tuning results 等到程序结束，我们可以查看当前最优的参数和结果：
xgb_BO.max
xgb_BO.res

xgb_BO_max = pd.DataFrame(xgb_BO.max).T




#######################################
#### retrain model with tuned parameters
######################################
params = xgb_BO_max.iloc[1].to_dict()

# xgboost.train()利用param列表设置模型参数
# xgboost.XGBClassifier()利用函数参数设置模型参数。 用途是一样的

best_xgb_iteration = 1500  # 前面num_boost_round = 1500

clf_train = xgb.XGBClassifier(learning_rate=0.01
                              , n_estimators=best_xgb_iteration
                              , max_depth=int(params['max_depth'])
                              , min_child_weight=params['min_child_weight']
                              , subsample=params['subsample']
                              , colsample_bytree=params['colsample_bytree']
                              , gamma=params['gamma']
                              , seed=1441
                              , nthread=-1
                              , scale_pos_weight=1
                              , eval_metric='auc'

                              )

params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.01,
          "max_depth": 4,
          "subsample": 0.5115777,
          "colsample_bytree":  0.509529,
          "min_child_weight": 13.93988,
          "gamma":  0.335555,
          "silent": 1,
          "seed": 1441,
          "eval_metric": "auc"
          }

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
num_boost_round = 1500

gbm2 = xgb.train(params,
                dtrain,
                num_boost_round,
                evals = watchlist,
                early_stopping_rounds = 50)

'''
clf = XGBClassifier(
silent=0 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
nthread=4,# cpu 线程数 默认最大
learning_rate= 0.3, # 如同学习率
min_child_weight=1, # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，
# 对正负样本不均衡时的0-1分类而言, 假设 h 在0.01附近，min_child_weight 为1意味着叶子节点中最少需要包含100个样本。
# 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易overfitting
max_depth=6, # 构建树的深度，越大越容易overfitting
gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
subsample=1, # 随机采样训练样本 训练实例的子采样比
max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
colsample_bytree=1, # 生成树时进行的列采样 
reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易overfitting
reg_alpha=0, # L1 正则项参数
scale_pos_weight=1, # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛,平衡正负权重
objective= 'multi:softmax', # 多分类的问题 指定学习任务和相应的学习目标
num_class=10, # 类别数，多分类与multisoftmax并用
n_estimators=100, # 树的个数
seed=1000 #随机种子
#eval_metric= 'auc'
'''



########################
### Model Evaluation ###
########################

### Feature importance

importance2 = gbm.get_fscore()

df_importance2 = pd.DataFrame.from_dict(importance2,orient='index').reset_index()
df_importance2.rename({0:'fscore','index':'feature'},axis=1,inplace=True)
df_importance2['fscore'] = df_importance2['fscore'] / df_importance2['fscore'].sum()
df_importance2.sort_values(['fscore'], ascending=False, inplace=True)
df_importance2

plt.figure(figsize=(32, 32))
df_importance[:20].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb2.png')



############################
### Validate on test data
y_pred = gbm.predict(dtest)
print (y_pred.max(), y_pred.min(), y_pred.mean())   # 0.71161723 0.008794592 0.13552907


### ROC Curve
idx_0, thresholds_0, idx_1, thresholds_1, idx_2, thresholds_2 = \
draw_ROC(gbm2, dtrain, dvalid, dtest, y_train, y_valid, y_test)

plt.savefig('roc_xgb_retrain_autotuned.png')
plt.show()
# train高，test低，过拟合


### Thresholds
plt.figure(figsize = (8,8))
plt.plot(thresholds_0, idx_0, label = "ROC curve - validation", color= 'r')
plt.plot(thresholds_1, idx_1, label = "ROC curve - train", color= 'b')
plt.plot(thresholds_2, idx_2, label = "ROC curve - test", color= 'g')
plt.xlabel('Thresholds')
plt.ylabel("TPR - FPR")


def max_threshold( x_feature, y_feature):
    y_max = max(y_feature)
    x_max = x_feature[ y_feature.argmax()] # Find the x value corresponding to the max y value
    return(x_max, y_max )

print ('validation:',max_threshold(thresholds_0, idx_0))
print ('train:',max_threshold(thresholds_1, idx_1))
print ('test:',max_threshold(thresholds_2, idx_2))



from sklearn.metrics import precision_recall_fscore_support
for thrd in [0, 0.1, 0.14, 0.15, 0.2, 0.25, 0.5]:
    test_results = []
    for prob in y_pred:
        if prob > thrd:
            test_results.append(1)
        else:
            test_results.append(0)
    print ("threshold:", thrd, precision_recall_fscore_support(y_test, test_results, pos_label=1, average='binary'))





