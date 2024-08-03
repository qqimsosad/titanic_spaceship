import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression



class FeatureEngineering:

    def __init__(self):
        pass

    def create_age_groups(self, df):
        df['Age_group']=np.nan
        df.loc[df['Age']<=12,'Age_group']='Age_0-12'
        df.loc[(df['Age']>12) & (df['Age']<18),'Age_group']='Age_13-17'
        df.loc[(df['Age']>=18) & (df['Age']<=25),'Age_group']='Age_18-25'
        df.loc[(df['Age']>25) & (df['Age']<=30),'Age_group']='Age_26-30'
        df.loc[(df['Age']>30) & (df['Age']<=50),'Age_group']='Age_31-50'
        df.loc[df['Age']>50,'Age_group']='Age_51+'
        return df

    def sum_expenditure(self, df, exp_feats):
        df['Expenditure'] = df[exp_feats].sum(axis=1)
        df['No_spending'] = (df['Expenditure'] == 0).astype(int)
        return df

    def extract_group_info(self, df):
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        return df

    def calculate_group_size(self, df, other_df):
        combined = pd.concat([df['Group'], other_df['Group']])
        df['Group_size'] = df['Group'].map(combined.value_counts())
        return df

    def solo_traveler_feature(self, df):
        df['Solo'] = (df['Group_size'] == 1).astype(int)
        return df

    def handle_cabin_info(self, df):
        df['Cabin'].fillna('Z/9999/Z', inplace=True)
        df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
        df['Cabin_number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
        df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
        
        # Replacing placeholders with NaN
        df.loc[df['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
        df.loc[df['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
        df.loc[df['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan
        df.drop('Cabin', axis=1, inplace=True)
        return df

    def handle_family_info(self, df, other_df):
        df['Name'].fillna('Unknown Unknown', inplace=True)
        other_df['Name'].fillna('Unknown Unknown', inplace=True)

        df['Surname']= df['Name'].str.split().str[-1]
        other_df['Surname']= other_df['Name'].str.split().str[-1]
   
        combined = pd.concat([df['Surname'], other_df['Surname']])

        df['Family_size'] = df['Surname'].map(combined.value_counts())
        other_df['Family_size'] = other_df['Surname'].map(combined.value_counts())
        
        # Handling Unknown and large family sizes
        df.loc[df['Surname'] == 'Unknown', 'Surname'] = np.nan
        df.loc[df['Family_size'] > 100, 'Family_size'] = np.nan
        other_df.loc[other_df['Surname'] == 'Unknown', 'Surname'] = np.nan
        other_df.loc[other_df['Family_size'] > 100, 'Family_size'] = np.nan


        df.drop('Name', axis=1, inplace=True)
        other_df.drop('Name', axis=1, inplace=True)

        return df,other_df


def preprocess_data(X, X_test):
    """
    對數據進行前處理，包括數值型數據的標準化和類別型數據的獨熱編碼。
    :param X: 訓練數據集 DataFrame
    :param X_test: 測試數據集 DataFrame
    :return: 處理後的訓練和測試數據集
    """
    # 確定數值型和類別型特徵
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # 數值型數據標準化
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # 類別型數據獨熱編碼
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False))])

    # 組合前處理
    ct = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)],
        remainder='passthrough')

    # 應用前處理
    X_preprocessed = ct.fit_transform(X)
    # X_test_preprocessed = ct.transform(X_test)

    return X_preprocessed

train = pd.read_csv('train.csv') #(8693, 14)
test = pd.read_csv('test.csv') #(4277, 13)
print(train.head())

fe = FeatureEngineering()
exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

#對訓練集和測試集進行特徵工程
train = fe.create_age_groups(train)
test = fe.create_age_groups(test)
print(train.head())


#exp_feats 是一個包含消費相關特徵名稱的列表 統計消費金額以及是否消費
train = fe.sum_expenditure(train, exp_feats)
test = fe.sum_expenditure(test, exp_feats)
print(train.head())


######要一起處理
# PassengerId是一個由乘客編號和組編號組成的字符串 這裡提取組編號
train = fe.extract_group_info(train)
test = fe.extract_group_info(test)
print(train.head())

# 計算每個組的人數 
train = fe.calculate_group_size(train, test)
test = fe.calculate_group_size(test, train)
print(train.head())

# 創建一個特徵來標識是否為單獨旅行者
train = fe.solo_traveler_feature(train)
test = fe.solo_traveler_feature(test)
print(train.head())
######

#處理客艙信息 1拆成3
train = fe.handle_cabin_info(train)
test = fe.handle_cabin_info(test)
print(train.head())

train,test = fe.handle_family_info(train, test)
print(train.head())

print(train.shape)
print(test.shape)

train.to_csv('train_feature_engine.csv', index=False)
test.to_csv('test_feature_engine.csv', index=False)