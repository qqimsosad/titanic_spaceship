import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def observe_missing_values(df):
    print("觀察缺失值分佈")
    print(df.isna().sum())
    print("\n")

def observe_data_variety(df):
    print("觀察資料種類")
    print(df.nunique())
    print("\n")

def observe_data_types(df):
    print("觀察資料型態")
    print(df.dtypes)
    print("\n")


class DataExplorer:
    def __init__(self, data):
        self.data = data

    def plot_pie_chart(self, column, figsize=(5,5)):
        plt.figure(figsize=figsize)
        self.data[column].value_counts().plot.pie(explode=[0.01,0.01], autopct='%1.1f%%',textprops={'fontsize':16})
        plt.show()

    def plot_histogram(self, column, binwidth=1, figsize=(13,7)):
        plt.figure(figsize=figsize)
        sns.histplot(data=self.data, x=column, hue='Transported', binwidth=binwidth, kde=True)
        plt.title(f'{column} distribution')
        plt.xlabel(f'{column} (years)')
        plt.show()

    def plot_expenditure_features(self, features, figsize=(10,20)):
        fig = plt.figure(figsize=figsize)
        for i, var_name in enumerate(features):
            ax = fig.add_subplot(5, 2, 2*i + 1)
            sns.histplot(data=self.data, x=var_name, bins=30, kde=False, hue='Transported')
            ax.set_title(var_name)

            ax = fig.add_subplot(5, 2, 2*i + 2)
            sns.histplot(data=self.data, x=var_name, bins=30, kde=True, hue='Transported')
            plt.ylim([0, 100])
            ax.set_title(var_name)
        fig.tight_layout()
        plt.show()

    def plot_categorical_features(self, features, figsize=(10,16)):
        fig = plt.figure(figsize=figsize)
        for i, var_name in enumerate(features):
            ax = fig.add_subplot(4, 1, i+1)
            sns.countplot(data=self.data, x=var_name, hue='Transported')
            ax.set_title(var_name)
        fig.tight_layout()
        plt.show()


train = pd.read_csv('train.csv') #(8693, 14)
test = pd.read_csv('test.csv') #(4277, 13)
print(train.head())
observe_missing_values(train)
# observe_data_variety(train)
# observe_data_types(train)
explorer = DataExplorer(train)

# 繪製s餅圖 占比T or F大約占50%
# explorer.plot_pie_chart('Transported')

# 繪製年齡直方圖
# explorer.plot_histogram('Age')

# 繪製支出特徵直方圖 觀察花錢的分佈
# exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# explorer.plot_expenditure_features(exp_feats)


# 繪製類別特徵計數圖 觀察其他類別分佈
# cat_feats = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
# explorer.plot_categorical_features(cat_feats)


