
# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, classification_report , f1_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from boruta import BorutaPy
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.metrics import fbeta_score
from time import time

df = pd.read_csv(r"C:\Users\ron\Documents\KUL Master informatiemanagement/Master thesis/data_ml.csv")

cat_vars = ['CUSTOMER_TYPE.x','RISK_RATING.x','PEP_FLAG','OFFSHORE_FLAG',
            'BUSINESS_TYPE_DESCRIPTION','ADDRESS_COUNTRY_CODE',
            'NATIONALITY_COUNTRY_CODE','PARENT_CHILD_FLAG','REGISTRATION_COUNTRY_CODE',
            'DOMESTIC_PEP_CODE','DOMESTIC_PEP_SOURCE_CODE','FOREIGN_PEP_CODE',
            'FOREIGN_PEP_SOURCE_CODE','ADDRESS_COUNTRY_REGION',
            'ADDRESS_COUNTRY_RISK_WEIGHT','CUST_TYPE_ALTERNATIVE_DESCR','EVENT_DATE']

num_vars = ['TTL_AMOUNT_MT103_202COV_TXN',
           'TTL_COUNT_MT103_202COV_TXN', 'TTL_AMOUNT_MT103_CREDIT',
           'TTL_COUNT_MT103_CREDIT', 'TTL_AMOUNT_MT202COV_CREDIT',
           'TTL_COUNT_MT202COV_CREDIT', 'TTL_AMOUNT_MT103_DEBIT',
           'TTL_COUNT_MT103_DEBIT', 'TTL_AMOUNT_MT202COV_DEBIT',
           'TTL_COUNT_MT202COV_DEBIT', 'MAX_AMOUNT_MT103_202COV_TXN',
           'MAX_AMOUNT_MT103_202COV_THR', 'TTL_AMT_PM_MT103_202COV',
           'AVG_AMT_3M_MT103_202COV', 'PERCNT_AMT_EXCEED_PM_3M',
           'THR_AMT_EXCEED_PM_3M', 'AMOUNT_MT103_202COV_M1',
           'AMOUNT_MT103_202COV_M2', 'AMOUNT_MT103_202COV_M3',
           'TTL_CNT_PM_MT103_202COV', 'AVG_CNT_3M_MT103_202COV',
           'PERCNT_CNT_EXCEED_PM_3M', 'THR_CNT_EXCEED_PM_3M',
           'COUNT_MT103_202COV_M1', 'COUNT_MT103_202COV_M2',
           'COUNT_MT103_202COV_M3', 'CLIENT_ACC_KNOWN_FOR_MONTH']

#### PIPELINES ####

### FEATURE ENGINEERING


numeric_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', RobustScaler())
                                        ]
                               )

categorical_transformer = Pipeline(steps = [('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
                                            ]
                                   )

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_vars),
                                               ('cat', categorical_transformer, cat_vars)
                                               ]
                                 )
pipe = Pipeline(steps=[('preprocessor', preprocessor)
                       ]
                )

### LABEL DEFINITION

## Binary Labels
cond_b1 = (df['ALERT_STATUS'] == 'Unsuspicious Alert') | \
          (df['ALERT_STATUS'] == 'Unsuspicious Case') | \
          (df['ALERT_STATUS'] == 'Approve Unsuspicious') 
              
y_b1 = np.where(cond_b1, 0, 1)


## Data Split
X_tr_b1, X_ts_b1, y_tr_b1, y_ts_b1 = train_test_split(df.drop(columns = 'ALERT_STATUS'), 
                                                      y_b1,
                                                      stratify = y_b1,
                                                      test_size=0.33,
                                                      random_state=123)


X_tr_pp_b1 = pipe.fit_transform(X_tr_b1)
X_ts_pp_b1 = pipe.transform(X_ts_b1)

variable_names = np.array(list(pipe['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(cat_vars)) + \
                          num_vars)

  
#### RESAMPLING ####
ada = ADASYN(random_state=123)
X_res, y_res = ada.fit_resample(X_tr_pp_b1, y_tr_b1) 

#### FEATURE SELECTION ####

boruta_rf = RandomForestClassifier(n_jobs=-1, n_estimators = 500)

# define Boruta feature selection method
feat_selector = BorutaPy(boruta_rf, verbose=2, random_state=123,
                         alpha=0.05, perc = 100, max_iter = 20, two_step = False)

feat_selector.fit(X_res.toarray(), y_res)

X_tr_res_bor = feat_selector.transform(X_res.toarray())
X_ts_res_bor = feat_selector.transform(X_ts_pp_b1)


# check selected features - first 5 features are selected
variable_names[feat_selector.support_]

#### MODELING ####
#Step 7a. Hyper parameter tuning DT
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_tr_res_bor, y_res)

print(accuracy_score(y_ts_b1, dt.predict(X_ts_res_bor)))
print(recall_score(y_ts_b1, dt.predict(X_ts_res_bor), average='macro'))
target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, dt.predict(X_ts_res_bor), target_names=target_names))

#Find the optimal hyperparamaters
dt = DecisionTreeClassifier(random_state=123)
param_grid = {'max_depth': list(np.linspace(1, 33, 16, dtype = int)) + [None],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [2,3,4,5],
              'ccp_alpha': list(np.linspace(0, 1, 50, dtype = float))}

pprint(param_grid)

random_search = RandomizedSearchCV(dt, 
                                   param_distributions= param_grid,
                                   scoring = 'f1_macro',
                                   cv = 5,
                                   n_iter= 25,
                                   random_state = 123)

#Step 7b. Hyper parameter tuning RF
rf = RandomForestClassifier(random_state=123)
rf.fit(X_tr_res_bor, y_res)

print(accuracy_score(y_ts_b1, rf.predict(X_ts_res_bor)))
print(recall_score(y_ts_b1, rf.predict(X_ts_res_bor), average='macro'))
target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, rf.predict(X_ts_res_bor), target_names=target_names))

#Find the optimal hyperparameters
rf = RandomForestClassifier(random_state=123)
param_grid = {'n_estimators': list(np.linspace(1, 200, 100, dtype = int)),
              'max_depth': list(np.linspace(1, 33, 16, dtype = int)) + [None],
              'criterion': ['entropy', 'gini'],
              'min_samples_split': [2,3,4,5],
              'ccp_alpha': list(np.linspace(0, 1, 50, dtype = float))}

pprint(param_grid)

random_search = RandomizedSearchCV(rf, 
                                   param_distributions= param_grid,
                                   scoring = 'f1_macro',
                                   cv = 5,
                                   n_iter= 25,
                                   random_state = 123)

#Step 7b. Hyper parameter tuning LR
lr = LogisticRegression(random_state=123)
lr.fit(X_tr_res_bor, y_res)

print(accuracy_score(y_ts_b1, lr.predict(X_ts_res_bor)))
print(recall_score(y_ts_b1, lr.predict(X_ts_res_bor), average='macro'))
target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, lr.predict(X_ts_res_bor), target_names=target_names))

#Hyperparameter tuning
lr = LogisticRegression(random_state=123)
param_grid = {'penalty': ['l2','l1', None],
              'solver': ['lbfgs', 'liblinear'],
              'C': [float(x) for x in np.linspace(0, 1, num = 20)],
              'multi_class': ['auto','ovr','multinomial']}

pprint(param_grid)

random_search = RandomizedSearchCV(lr, 
                                   param_distributions= param_grid,
                                   scoring = 'f1_macro',
                                   cv = 5,
                                   n_iter=25,
                                   random_state = 123)

#### CROSS-VALIDATION #### can be used for all models because they have same variable 'random_search'
start = time()
random_search.fit(X_tr_res_bor, y_res)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), 5))
random_search.best_params_


###FINAL RESULTS WITH OPT HYPERPARAMETERS###

#Step 8a: New training with opt hyperparameters DT
dt_opt = DecisionTreeClassifier(max_depth=15, max_features='auto',criterion='gini',ccp_alpha=0,random_state=123)
dt_opt.fit(X_tr_res_bor, y_res)
y_pred_dt = dt_opt.predict(X_ts_res_bor)

print(accuracy_score(y_ts_b1, y_pred_dt))
print(recall_score(y_ts_b1, y_pred_dt, average='macro'))

target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, y_pred_dt, target_names=target_names))
print(f1_score(y_ts_b1, y_pred_dt, average='macro'))
print(f1_score(y_ts_b1, y_pred_dt, average='weighted'))
print(fbeta_score(y_ts_b1, y_pred_dt, average='macro', beta=0.5))
print(fbeta_score(y_ts_b1, y_pred_dt, average='weighted', beta=0.5))

#Step 8b: New training with opt hyperparameters RF
rf_opt = RandomForestClassifier(n_estimators=61, max_depth= 22, min_samples_split=3,criterion= 'entropy',ccp_alpha=0.02, random_state=123)
rf_opt.fit(X_tr_res_bor, y_res)
y_pred_rf = rf_opt.predict(X_ts_res_bor)

print(accuracy_score(y_ts_b1, y_pred_rf))
print(recall_score(y_ts_b1, y_pred_rf, average='macro'))

target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, y_pred_rf, target_names=target_names))
print(f1_score(y_ts_b1, y_pred_rf, average='macro'))
print(f1_score(y_ts_b1, y_pred_rf, average='weighted'))
print(fbeta_score(y_ts_b1, y_pred_rf, average='macro', beta=0.5))
print(fbeta_score(y_ts_b1, y_pred_rf, average='weighted', beta=0.5))


#Step 8c: New training with opt hyperparameters LR
lr_opt = LogisticRegression(C=0.4737,solver = 'liblinear',penalty='l1',n_jobs=-1,random_state=123)
lr_opt.fit(X_tr_res_bor, y_res)
y_pred_lr = lr_opt.predict(X_ts_res_bor)

print(accuracy_score(y_ts_b1, y_pred_lr))
print(recall_score(y_ts_b1, y_pred_lr, average='macro'))

target_names = ['Unsuspicious', 'Suspicious']
print(classification_report(y_ts_b1, y_pred_lr, target_names=target_names))
print(f1_score(y_ts_b1, y_pred_lr, average='macro'))
print(f1_score(y_ts_b1, y_pred_lr, average='weighted'))
print(fbeta_score(y_ts_b1, y_pred_lr, average='macro', beta=0.5))
print(fbeta_score(y_ts_b1, y_pred_lr, average='weighted', beta=0.5))