import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
train_data = pd.read_csv('train_feature_engine.csv')
test_data = pd.read_csv('test_feature_engine.csv')

# 在進行任何處理前，保存 PassengerId
test_passenger_ids = test_data['PassengerId'].copy()

# Identifying numerical and categorical columns
num_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
cat_columns = train_data.select_dtypes(include=['object', 'bool']).columns.drop('Transported')  # Exclude the label column

# Imputing missing values
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

train_data[num_columns] = num_imputer.fit_transform(train_data[num_columns])
test_data[num_columns] = num_imputer.transform(test_data[num_columns])

train_data[cat_columns] = cat_imputer.fit_transform(train_data[cat_columns])
test_data[cat_columns] = cat_imputer.transform(test_data[cat_columns])

# Encoding categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_encoded = encoder.fit_transform(train_data[cat_columns])
test_encoded = encoder.transform(test_data[cat_columns])

train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(cat_columns))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(cat_columns))

# Combining encoded features with original data
train_data = train_data.drop(columns=cat_columns).reset_index(drop=True)
test_data = test_data.drop(columns=cat_columns).reset_index(drop=True)

train_data = pd.concat([train_data, train_encoded_df], axis=1)
test_data = pd.concat([test_data, test_encoded_df], axis=1)

print(train_data.shape)
print(test_data.shape)

# Now your data is ready for training models

# 分割數據為訓練集和驗證集
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 初始化模型，設置日誌輸出
random_forest = RandomForestClassifier()
lgbm = LGBMClassifier()
catboost = CatBoostClassifier(verbose=100)
xgboost = XGBClassifier()

# 訓練模型

print("\nTraining Random Forest...")
random_forest.fit(X_train, y_train)

print("\nTraining LGBM...")
lgbm.fit(X_train, y_train)

print("\nTraining CatBoost...")
catboost.fit(X_train, y_train)

print("\nTraining XGBoost...")
xgboost.fit(X_train, y_train)



# 在驗證集上進行預測
rf_predictions = random_forest.predict(X_val)
lgbm_predictions = lgbm.predict(X_val)
catboost_predictions = catboost.predict(X_val)
xgboost_predictions = xgboost.predict(X_val)


# 將預測結果從字符串轉換為布林值
catboost_predictions = [True if x == 'True' else False for x in catboost_predictions]


# 計算準確率
rf_accuracy = accuracy_score(y_val, rf_predictions)
lgbm_accuracy = accuracy_score(y_val, lgbm_predictions)
catboost_accuracy = accuracy_score(y_val, catboost_predictions)
xgboost_accuracy = accuracy_score(y_val, xgboost_predictions)


print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"LGBM Accuracy: {lgbm_accuracy}")
print(f"CatBoost Accuracy: {catboost_accuracy}")
print(f"XGBoost Accuracy: {xgboost_accuracy}")


# 使用測試數據進行預測
rf_test_predictions = random_forest.predict(test_data)
lgbm_test_predictions = lgbm.predict(test_data)
catboost_test_predictions = catboost.predict(test_data)
xgboost_test_predictions = xgboost.predict(test_data)

# 在創建提交文件時使用保存的 PassengerId
rf_submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Transported': rf_test_predictions})
lgbm_submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Transported': lgbm_test_predictions})
catboost_submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Transported': catboost_test_predictions})
xgboost_submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Transported': xgboost_test_predictions})

# 保存預測結果到 CSV 文件
rf_submission.to_csv('rf_submission.csv', index=False)
lgbm_submission.to_csv('lgbm_submission.csv', index=False)
catboost_submission.to_csv('catboost_submission.csv', index=False)
xgboost_submission.to_csv('xgboost_submission.csv', index=False)