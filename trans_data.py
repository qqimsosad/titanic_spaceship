import pandas as pd

# 讀取 XGBoost 預測結果檔案
xgboost_predictions = pd.read_csv('xgboost_submission.csv')

# 將預測結果轉換為布爾型
xgboost_predictions['Transported'] = xgboost_predictions['Transported'].astype(bool)

# 保存轉換後的預測結果
xgboost_predictions.to_csv('xgboost_predictions_converted.csv', index=False)
