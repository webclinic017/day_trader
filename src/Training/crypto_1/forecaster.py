import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import zscore
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import multiprocessing as mp
import os
import itertools

class Forecaster():

    coin_type: int
    df: pd.DataFrame

    train_X: np.array
    train_y: np.array

    params: dict
    model: object

    min_interval: int
    chunks: list 

    def __init__(self, coin_type: int, min_interval: int, params: dict = {}) -> None:
        
        self.coin_type = coin_type
        self.params = params
        self.min_interval = min_interval

        if not os.path.isdir("forecast_models"):
            os.mkdir("forecast_models")
        
    def load(self):

        self.model = pickle.load(open(f"forecast_models/model_{self.coin_type}", "rb"))
        self.perform_preprocessing()
    
    def load_chunks(self):
        
        self.chunks = pickle.load(open(f"forecast_models/chunks_{self.coin_type}.pkl", "rb"))

    def perform_preprocessing(self):
        
        if os.path.isfile(f"forecast_models/chunks_{self.coin_type}.pkl"):
            print(f"Skipping preprocessing because already done for: {self.coin_type}")
            self.load_chunks()
            return

        df = pd.read_csv("data/train.csv")

        dfs = [x for _, x in df.groupby('Asset_ID')]   
        df = dfs[self.coin_type]                                            # Setting the coin

        df = df.fillna(method='ffill')
    
        df['Count'] =  df['Count'].apply(np.log)
        df['Open'] = df['Open'].apply(np.log)
        df['High'] = df['High'].apply(np.log)
        df['Low'] = df['Low'].apply(np.log)
        df['Close'] = df['Close'].apply(np.log)
        df['Volume'] = df['Volume'].apply(np.log)
        df['VWAP'] = df['VWAP'].apply(np.log)

        df['Count'] = ((df['Count']-df['Count'].mean())/df['Count'].std())
        df['Open'] = ((df['Open']-df['Open'].mean())/df['Open'].std())
        df['High'] = ((df['High']-df['High'].mean())/df['High'].std())
        df['Low'] = ((df['Low']-df['Low'].mean())/df['Low'].std())
        df['Close'] = ((df['Close']-df['Close'].mean())/df['Close'].std())
        df['Volume'] = ((df['Volume']-df['Volume'].mean())/df['Volume'].std())
        df['VWAP'] = ((df['VWAP']-df['VWAP'].mean())/df['VWAP'].std())

        df = df.set_index('timestamp')
        df = df.reindex(range(df.index[0],df.index[-1]+60,60),method='pad') # padding missing values
        df = df.reset_index(level=['timestamp'])

        df['timestamp'] = df['timestamp'].apply(datetime.fromtimestamp)
        df['time_bucket'] = df['timestamp'].apply(lambda x: int(x.hour / 4))  # segmenting times into buckets depending on 4h groups throughout the day

        enc = OneHotEncoder(sparse=False)
        time_onehot = enc.fit_transform(df[['time_bucket']])
        t_hot = pd.DataFrame(time_onehot, columns=list(enc.categories_[0]))
        df.drop('time_bucket', axis=1, inplace=True)

        for i in range(6):
            df[f"Hour_{i}"] = t_hot[i]
       
        train = df[(df['timestamp'] > '2017-12-31') & (df['timestamp'] < '2021-01-01')]
        train.drop('timestamp', axis=1, inplace=True)

        self.train_X = train.drop(['Target', 'Asset_ID'], inplace=False, axis=1)

        N = len(self.train_X.index)
        self.train_X['grp'] = [int(i/60) for i in range(N)]
        self.train_X = self.train_X.groupby('grp').mean()

        for i in range(6):
            self.train_X[f"Hour_{i}"] = self.train_X[f"Hour_{i}"].apply(round)
        
        self.create_chunks()

    def create_chunks(self):
        
        df_array = self.train_X.to_numpy()
        chunks = []

        for i in range(len(df_array)):
            
            if i+10 < len(df_array):
                chunks.append(df_array[i:i+10])

        with open(f"forecast_models/chunks_{self.coin_type}.pkl", "wb") as f:
            pickle.dump(chunks, f)

        self.chunks = chunks

    def predict(self, chunk_num: list):

        return self.model.predict(self.chunks[chunk_num][-1])

    def train_model(self):

        self.perform_preprocessing()

        if not self.params:
            self.optimize()
        
        else:
            self.train_XGB()

    def optimize(self):
        
        params = { 'max_depth': [3,10,20],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000,],
           'colsample_bytree': [0.3, 0.7],
           'subsample': [0.2, 0.5, 1],
           'eta': [0.1, 0.5, 0.8],
           'min_child_weight': [100, 300, 500]
           }

        xgbr = XGBRegressor(seed = 40)
        
        model = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=4,
                   n_jobs=-1)
        
        model.fit(
            self.train_X, 
            self.train_y, 
            eval_metric="rmse", 
            eval_set=[(self.train_X, self.train_y), (self.test_X, self.test_y)], 
            verbose=True, 
            early_stopping_rounds = 10
        )

        print("Best parameters:", model.best_params_)

        pickle.dump(model, open(f"forecast_models/model_{self.coin_type}.pkl", "wb"))

    def plot_feature_importance(self, model: object):

        feature_importances = pd.DataFrame({'col': self.train_X.columns,'imp':model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='imp',ascending=False)
        x = list(feature_importances.col)
        y = list(feature_importances.imp)

        plt.bar(x, y)
        plt.show()
    
    def plot_performance(self, model):

        predicted_y = model.predict(self.test_X)

        x = [i for i in range(len(predicted_y))]
        plt.plot(x, self.test_y, color="blue")
        plt.plot(x, predicted_y, color="red")

        plt.show()


def train_XGB(params, coin_type: int):
        
        
        train_X, train_y, test_X, test_y = perform_preprocessing(coin_type)

        model = XGBRegressor(seed = 40, **params)
        
        model.fit(
            train_X, 
            train_y, 
            eval_metric="rmse", 
            eval_set=[(train_X, train_y), (test_X, test_y)], 
            verbose=True, 
            early_stopping_rounds = 10
        )

        pickle.dump(model, open(f"forecast_models/model_{coin_type}.pkl", "wb"))

def train_in_parralel():
    
    params = { 'max_depth': 10,
           'learning_rate': 0.03,
           'n_estimators': 500,
           'colsample_bytree': 0.4,
           'subsample': 0.3,
           'eta': 0.2,
           'min_child_weight': 300
           }


    pool = mp.Pool()

    for i in range(14):
        pool.apply_async(func=train_XGB, args=(params, i))
    
    pool.close()
    pool.join()

def create_chunks(train_X, closing, coin_type):
        
    print(f"Saving chunks for: {coin_type}")
    df_array = train_X.to_numpy()
    chunks = []

    for i in range(len(df_array)):
        
        if i+10 < len(df_array):
            chunks.append(df_array[i:i+10])

    with open(f"forecast_models/chunks_{coin_type}.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    with open(f"forecast_models/closing_{coin_type}.pkl", "wb") as f:
        pickle.dump(closing, f)
    



def perform_preprocessing(coin_type: int):
        
        print(f"starting: {coin_type}")

        df = pd.read_csv("data/train.csv")

        dfs = [x for _, x in df.groupby('Asset_ID')]   
        df = dfs[coin_type]                                            # Setting the coin

        df = df.fillna(method='ffill')

        df['Save_Close'] = df['Close']

        df['Count'] =  df['Count'].apply(np.log)
        df['Open'] = df['Open'].apply(np.log)
        df['High'] = df['High'].apply(np.log)
        df['Low'] = df['Low'].apply(np.log)
        df['Close'] = df['Close'].apply(np.log)
        df['Volume'] = df['Volume'].apply(np.log)
        df['VWAP'] = df['VWAP'].apply(np.log)

        df['Count'] = ((df['Count']-df['Count'].mean())/df['Count'].std())
        df['Open'] = ((df['Open']-df['Open'].mean())/df['Open'].std())
        df['High'] = ((df['High']-df['High'].mean())/df['High'].std())
        df['Low'] = ((df['Low']-df['Low'].mean())/df['Low'].std())
        df['Close'] = ((df['Close']-df['Close'].mean())/df['Close'].std())
        df['Volume'] = ((df['Volume']-df['Volume'].mean())/df['Volume'].std())
        df['VWAP'] = ((df['VWAP']-df['VWAP'].mean())/df['VWAP'].std())

        df = df.set_index('timestamp')
        df = df.reindex(range(df.index[0],df.index[-1]+60,60),method='pad') # padding missing values
        df = df.reset_index(level=['timestamp'])

        df['timestamp'] = df['timestamp'].apply(datetime.fromtimestamp)
        df['time_bucket'] = df['timestamp'].apply(lambda x: int(x.hour / 4))  # segmenting times into buckets depending on 4h groups throughout the day

        enc = OneHotEncoder(sparse=False)
        time_onehot = enc.fit_transform(df[['time_bucket']])
        t_hot = pd.DataFrame(time_onehot, columns=list(enc.categories_[0]))
        df.drop('time_bucket', axis=1, inplace=True)

        for i in range(6):
            df[f"Hour_{i}"] = t_hot[i]
       
        train = df[(df['timestamp'] > '2017-12-31') & (df['timestamp'] < '2021-01-01')]
        train.drop('timestamp', axis=1, inplace=True)

        train_X = train.drop(['Target', 'Asset_ID'], inplace=False, axis=1)

        N = len(train_X.index)
        train_X['grp'] = [int(i/60) for i in range(N)]
        train_X = train_X.groupby('grp').mean()


        for i in range(6):
            train_X[f"Hour_{i}"] = train_X[f"Hour_{i}"].apply(round)
        
        closing = list(train_X['Save_Close'])
        create_chunks(train_X, closing, coin_type)

def create_chunks_parallel():

    pool = mp.Pool()

    for i in range(14):
        pool.apply_async(func=perform_preprocessing, args=(i,))
    
    pool.close()
    pool.join()


def main():

    create_chunks_parallel()
    

if __name__ == '__main__':
    main()