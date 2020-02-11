import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
import time

# ----------载入数据------------
data_all = pd.read_csv('data_all.csv', index_col=0)
# ----------划分数据------------
train_x = data_all.drop(['is_leave_mattress', 'is_body_move'], axis=1)
train_y = data_all['is_body_move']
# train_y = data_all['is_leave_mattress']
auc_avg = 0
f1_avg = 0

# ----------lightgbm 5折交叉验证------------
for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=2020 + 100 * i)

    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_eval = lgb.Dataset(X_test, Y_test)

    params = {'learning_rate': 0.01,
              'metric': ['auc', 'binary_logloss'],
              'boosting': 'gbdt',
              'objective': 'binary',
              'nthread': 8,
              'num_leaves': 8,
              'colsample_bytree': 0.7,
              'bagging_fraction': 0.8,
              'bagging_freq': 10,
              'seed': 2020,
              }
    lgb_model = lgb.train(params, train_set=lgb_train, num_boost_round=10000, valid_sets=lgb_eval, verbose_eval=50,
                          early_stopping_rounds=100)
    pred = lgb_model.predict(X_test)
    auc_avg += roc_auc_score(Y_test, pred)
    print('TRAIN SET auc ', roc_auc_score(Y_test, pred))
    f1_ans = [1 if j >= 0.5 else 0 for j in pred]
    f1_avg += f1_score(Y_test, f1_ans)
    print('TRAIN SET F1 ', f1_score(Y_test, f1_ans))

    fi = pd.DataFrame()
    fi['name'] = lgb_model.feature_name()
    fi['score'] = lgb_model.feature_importance()
    print(fi.sort_values(by=['score'], ascending=False))

print('平均auc:{},平均f1:{}'.format(auc_avg / 5, f1_avg / 5))

# ------------lightgbm 5折交叉验证的平均auc:0.929384，平均f1: 0.73300528-----------强特为 diff_kurt、diff_skew、diff_var----还可以做xgboost、catboost,我懒得做了

# ------------logisticregression 5 折交叉验证-------------------------
start = time.time()
from sklearn.linear_model import LogisticRegression

for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=2020 + 100 * i)
    clf = LogisticRegression(random_state=2020, n_jobs=-1, verbose=100, max_iter=1000).fit(X_train.values,
                                                                                           Y_train.values)
    pred = clf.predict_proba(X_test.values)[:, 1]
    auc_avg += roc_auc_score(Y_test.values, pred)
    print('TRAIN SET auc ', roc_auc_score(Y_test.values, pred))
    f1_ans = [1 if j >= 0.5 else 0 for j in pred]
    f1_avg += f1_score(Y_test.values, f1_ans)
    print('TRAIN SET F1 ', f1_score(Y_test.values, f1_ans))
print('平均auc:{},平均f1:{}'.format(auc_avg / 5, f1_avg / 5))
print('总用时{}s'.format(time.time() - start))

# -----------logisticregression5折交叉验证的评价auc:0.8999122971990602,平均f1:0.6606449801714701-----------

# -----------randomforest 5 折交叉验证----------------------
start = time.time()
from sklearn.ensemble import RandomForestClassifier

for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=2020 + 100 * i)
    clf = RandomForestClassifier(
        n_estimators=50,
        criterion='gini',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=2020,
        verbose=0,
        warm_start=False,
        class_weight='balanced'
    )
    clf.fit(X=X_train, y=Y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    auc_avg += roc_auc_score(Y_test, pred)
    print('TRAIN SET auc ', roc_auc_score(Y_test, pred))
    f1_ans = [1 if j >= 0.5 else 0 for j in pred]
    f1_avg += f1_score(Y_test, f1_ans)
    print('TRAIN SET F1 ', f1_score(Y_test, f1_ans))
    feature_importances_sorted = sorted(zip(X_train.columns, clf.feature_importances_),
                                        key=lambda x: x[1], reverse=True)
    feature_importances = pd.DataFrame([list(f) for f in feature_importances_sorted],
                                       columns=["features", "importance"])
print('平均auc:{},平均f1:{}'.format(auc_avg / 5, f1_avg / 5))
print('总用时{}s'.format(time.time() - start))

# -----------randomforest5折交叉验证的平均auc:0.9172766770274443,平均f1:0.5674216087219494----强特为diff_mad,diff_std,peak

# -----------svm 5 折交叉验证--------------
start = time.time()
from sklearn.svm import SVC

for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=2020 + 100 * i)
    params = {
        "C": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "gamma": "auto",
        "coef0": 0.0,
        "tol": 0.001,
        "cache_size": 4000,
        "verbose": True,
        "max_iter": -1,
        "probability": True,
        "random_state": 2020
    }
    clf = SVC(**params)
    clf.fit(X=X_train.values, y=Y_train.values)
    pred = clf.predict_proba(X_test.values)[:, 1]
    auc_avg += roc_auc_score(Y_test.values, pred)
    print('TRAIN SET auc ', roc_auc_score(Y_test.values, pred))
    f1_ans = [1 if j >= 0.5 else 0 for j in pred]
    f1_avg += f1_score(Y_test.values, f1_ans)
    print('TRAIN SET F1 ', f1_score(Y_test.values, f1_ans))
print('平均auc:{},平均f1:{}'.format(auc_avg / 5, f1_avg / 5))
print('总用时{}s'.format(time.time() - start))

# 辣鸡svm是真的慢，跑不动-----------------------------------

# ----------最后还要测试一个RNN or ?.LSTM TODO --------------
