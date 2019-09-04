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

