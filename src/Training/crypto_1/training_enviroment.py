import random
from re import M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from os.path import exists
from datetime import datetime
from dataclasses import dataclass
import itertools
import os
import forecaster
from sklearn.preprocessing import OneHotEncoder
import sys

class TrainingEnviroment:
    
    chunks: list
    curr_chunk: int
    num_chunks: int
    buy_price: int
    curr_money: int
    closing_prices: list
    max_profit: int
    goal_profit: int
    min_interval: int
    month: int
    num_holdings: int = 0

    closing_prices: list
    support_coins: dict
    test_coin: int
    
    train_X: pd.DataFrame
    train_y: pd.DataFrame
    test_X: pd.DataFrame
    test_y: pd.DataFrame

    def __init__(self, test_coin: int, support_coins: dict, train: bool, min_interval: int = 60) -> None:
        """ Base level initialier function for the class.
        Takes list of chunks and closing_prices to initialize
        the overall enviroment
        
        Arguments:
            chunks (list): List of segement chunsk

            closing_prices (list): List of correlated prices 

        Return:
            None
        
        Side Effects:
            None
        """
        
        self.test_coin = test_coin
        self.support_coins = support_coins

        self.min_interval = min_interval
        self.month = int(40_320 / self.min_interval)     

        self.perform_preprocessing(train)
        self.create_forecasters()

        self.curr_chunk = random.randint(0,(len(self.chunks) - int(5000 / self.min_interval)))
        self.num_chunks = 0
        self.buy_price = -1
        self.curr_money = 1000
        self.prev_money = 1000
        self.max_profit = 1000
        self.goal_profit = int(1000 * self.weekly_return())
 
    def create_forecasters(self) -> None:
        """ Loops through the support coins and creates corresponding
        forecast objects for each

        Arguements:
            None

        Return: 
            None
        """

        for coin_num in self.support_coins:
            self.support_coins[coin_num] = forecaster.Forecaster(coin_num)
            self.support_coins[coin_num].perform_preprocessing()

    def perform_preprocessing(self, is_train: bool):
        
        if self.load_data(is_train):
            return

        df = pd.read_csv("data/train.csv")

        dfs = [x for _, x in df.groupby('Asset_ID')]   
        
        df = dfs[self.test_coin]                                            # Setting the coin
        del dfs

        df = df.fillna(method='ffill')
        df['old_Close'] = df['Close']

        df['Count'] =  df['Count'].apply(np.log)
        df['Close'] =  df['Close'].apply(np.log)
        df['Open'] = df['Open'].apply(np.log)
        df['High'] = df['High'].apply(np.log)
        df['Low'] = df['Low'].apply(np.log)
        df['Volume'] = df['Volume'].apply(np.log)
        df['VWAP'] = df['VWAP'].apply(np.log)

        df['Close'] = ((df['Close']-df['Close'].mean())/df['Close'].std())
        df['Count'] = ((df['Count']-df['Count'].mean())/df['Count'].std())
        df['Open'] = ((df['Open']-df['Open'].mean())/df['Open'].std())
        df['High'] = ((df['High']-df['High'].mean())/df['High'].std())
        df['Low'] = ((df['Low']-df['Low'].mean())/df['Low'].std())
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
        test = df[(df['timestamp'] >= '2021-01-01')]

        train.drop('timestamp', axis=1, inplace=True)
        test.drop('timestamp', axis=1, inplace=True)

        train_closing = train["old_Close"]
        test_closing = test["old_Close"]

        train.drop(['Target', 'Asset_ID', 'old_Close'], inplace=True, axis=1)
        test.drop(['Target', 'Asset_ID', 'old_Close'], inplace=True, axis=1)

        train.to_csv(f"forecast_models/train_{self.test_coin}.csv", encoding='utf-8', index=False)
        test.to_csv(f"forecast_models/test_{self.test_coin}.csv", encoding='utf-8', index=False)
        train_closing.to_csv(f"forecast_models/train_closing_{self.test_coin}.csv", encoding='utf-8', index=False)
        test_closing.to_csv(f"forecast_models/test_closing_{self.test_coin}.csv", encoding='utf-8', index=False)


        if is_train:
            self.create_chunks(train, train_closing)
        else:
            self.create_chunks(test, test_closing)
         
    def load_data(self, is_train: bool) -> None:
        """ Loads the previously created chunks from file

        Arguements:
            None

        Return:
            None
        """
        if is_train and os.path.isfile(f"forecast_models/train_{self.test_coin}.csv"):
            
            df = pd.read_csv(f"forecast_models/train_{self.test_coin}.csv")
            closing = pd.read_csv(f"forecast_models/train_closing_{self.test_coin}.csv")
            self.create_chunks(df, closing)
            return True
        
        elif not is_train and os.path.isfile(f"forecast_models/test_{self.test_coin}.csv"):

            df = pd.read_csv(f"forecast_models/test_{self.test_coin}.csv")
            closing = pd.read_csv(f"forecast_models/test_closing_{self.test_coin}.csv")
            self.create_chunks(df, closing)
            return True

        else:
            return False

    def create_chunks(self, df: pd.DataFrame, closing: pd.DataFrame) -> None:
        """ Function to segment the dataframe into overlapping
        2d array of 10 hours per chunk

        Arguements:
            X (pd.DataFrame): The dataframe to be chunked

        Return:
            None
        """

        N = len(df.index)
        
        df['grp'] = [int(i/self.min_interval) for i in range(N)]
        df = df.groupby('grp').mean()
        
        closing['grp'] = [int(i/self.min_interval) for i in range(N)]
        closing = closing.groupby('grp').mean()

        for i in range(6):
            df[f"Hour_{i}"] = df[f"Hour_{i}"].apply(round)

        df_array = df.to_numpy()
        closing_master = closing.to_numpy()

        chunks = []
        closing = []

        for i in range(len(df_array)):
            
            if i+10 < len(df_array):
                chunks.append(df_array[i:i+10])
                closing.append(closing_master[i+9][0])     

        self.chunks = chunks
        self.closing_prices = closing

    def weekly_return(self) -> int:
        """ Function to calculate the percentage gain by the coin
        over the coming week

        Arguements:
            None

        Return:
            chnage (int): Weekly percentile change in price
        """

        weekly_start_prices = self.closing_prices[self.curr_chunk]

        if self.curr_chunk+self.month < len(self.closing_prices):
            weekly_stop_prices = self.closing_prices[self.curr_chunk+self.month]
        else:
            weekly_stop_prices = self.closing_prices[len(self.closing_prices) - 1]

        change = (weekly_stop_prices/weekly_start_prices)

        if change < 1:
            change = 1.005
        else:
            change += 0.005
            
        return change
    
    def get_current_stock_price(self) -> float:
        """ Returns the current stock price for the chunk

        Arguements:
            None
        
        Return stock_price (int): The current stock price
        """

        return self.closing_prices[self.curr_chunk]

    def reset(self) -> None:
        """ Function to reset all state values of the class.
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """

        self.curr_chunk = random.randint(0,(len(self.chunks) - int(5000 / self.min_interval)))
        self.buy_price = -1
        self.curr_money = 1000
        self.prev_money = 1000
        self.max_profit = 1000
        self.goal_profit = int(1000 * self.weekly_return())
        self.num_chunks = 0

    def get_past_10_avg(self) -> int:
        """ Calculates the previous 10min of prices data from
        the SPY stock
        
        Arguments:
            None

        Return:
            average (int): The 10min average
        
        Side Effects:
            None
        """

        if self.curr_chunk > 9:
            past_10min = self.closing_prices[self.curr_chunk-10:self.curr_chunk]
        else:
            past_10min = self.closing_prices[0:self.curr_chunk]

        total = sum(past_10min)
        avg = total/len(past_10min)

        return avg

    def get_current_money(self) -> int:
        """ Calculates the current equity of the account
        by theorizing how much the stocks are worth if they were sold now
        and adding that to the cash held by the account.
        
        Arguments:
            None

        Return:
            current_money (int): The current money for the model
        
        Side Effects:
            None
        """
        

        if self.buy_price != -1:
            
            return round((self.curr_money * (self.closing_prices[self.curr_chunk] / self.buy_price)), 2)
        
        else:
            return self.curr_money
  
    def sell(self, current: int, decay: int) -> int:
        """ Caclualtes the reward for selling the current stock

        
        Arguments:
            current (int): The current price of the stock

            decay (int): The current decay of the training model 

        Return:
            reward (int): The profit or loss for selling the given stock
        
        Side Effects:
            None
        """
        
        if self.buy_price != -1:
            
            self.curr_money = round((self.curr_money * (current / self.buy_price)), 1)
            self.buy_price = -1

            return self.curr_money

        else:
            return 0

    def get_current_state(self) -> list:
        """ Creates combined array of the current prices and
        volumes of all stock, in combination with the price_list,
        the current money, how far into the week the model is and
        finally, the goal profit of the stock.
        
        Arguments:
            None

        Return:
            curr_state (list): the current state of the model
        
        Side Effects:
            None
        """
        
        curr_state = np.copy(self.chunks[self.curr_chunk])
        
        for coin_type in self.support_coins:
            curr_state = np.append(curr_state, self.support_coins[coin_type].predict(self.curr_chunk))
        
        curr_state = np.append(curr_state, self.buy_price)
        curr_state = np.append(curr_state, self.get_current_money())
        curr_state = np.append(curr_state, self.num_chunks)
        curr_state = np.append(curr_state, self.goal_profit)

        return curr_state

    def test_step_v2(self, action: int, prev_chunk: list, decay: int) -> int:
        """ Executes the decided action from the model. If 1,
        the model buys $100 of SPY stock. If 2, the model holds and 
        does nothing. If 3, the model sells all current holdings of the
        SPY stock.
        
        Arguments:
            action (int): The given action from the model

            prev_chunk (list): The list of the previous chunk

        Return:
            reward (int): The reward given for the model
        
        Side Effects:
            None
        """
       
        self.num_chunks += 1

        if self.curr_chunk < (len(self.chunks) -1):
 
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.buy_price == -1 :
                    self.buy_price = self.closing_prices[self.curr_chunk -1]      # Appending current price we bought the stock at for previous chunk 
                    self.num_holdings = 0

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_price == -1 and self.num_holdings > 4:
                    reward = -1
                
                self.num_holdings +=1
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                self.sell(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                reward = 0
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks % self.month == 0:        # Checking if we made 0.5 % for the week

                if self.get_current_money() < self.goal_profit:
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()
                    
                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()
                    
                    return -1, True, False              # Not ending because were testing

                else:
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 1, True, False

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward, False, False
        
        else:
            return 0, True, True
    
    def test_step(self, action: int, prev_chunk: list, decay: int) -> int:
        """ Executes the decided action from the model. If 1,
        the model buys $100 of SPY stock. If 2, the model holds and 
        does nothing. If 3, the model sells all current holdings of the
        SPY stock.
        
        Arguments:
            action (int): The given action from the model

            prev_chunk (list): The list of the previous chunk

        Return:
            reward (int): The reward given for the model
        
        Side Effects:
            None
        """
       
        self.num_chunks += 1

        if self.curr_chunk < (len(self.chunks) -1):
 
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.buy_price != -1:
                    self.buy_price = self.closing_prices[self.curr_chunk -1]       # Appending current price we bought the stock at for previous chunk 
                    self.num_holdings = 0

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN
                
                if self.buy_price == -1 and self.num_holdings > 4:
                    reward = -1
                
                self.num_holdings += 1
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                self.sell(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                reward = 0
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks > (self.month - 1) and self.num_chunks % self.month == 0:        # Checking if we made 0.5 % for the week

                if self.get_current_money() < self.goal_profit:
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()
                    
                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()
                    
                    return -1, False              # Not ending because were testing

                else:
                    
                    self.goal_profit = self.get_current_money() * self.weekly_return()

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 1, False

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward, False
        
        else:
            return 0, True

    def step(self, action, prev_chunk, decay) -> int: 
        """ Executes the decided action from the model. If 1,
        the model buys $100 of SPY stock. If 2, the model holds and 
        does nothing. If 3, the model sells all current holdings of the
        SPY stock.
        
        Arguments:
            action (int): The given action from the model

            prev_chunk (list): The list of the previous chunk

        Return:
            reward (int): The current reward given for the model
        
        Side Effects:
            None
        """
        
        self.num_chunks += 1

        if self.curr_chunk + 1 < int(len(self.chunks)):
 
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.buy_price == -1:
                    self.buy_price = self.closing_prices[self.curr_chunk -1]       # Appending current price we bought the stock at for previous chunk 
                    self.num_holdings = 0

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_price == -1 and self.num_holdings > 4:
                    reward = -1
                
                self.num_holdings +=1
                
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                self.sell(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks % self.month == 0:        # Checking if we made 1 % for the week

                if self.get_current_money() < self.goal_profit:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    return -1, True

                else:
                    
                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 1,  True

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward,  False
        
       
        else:
            self.curr_chunk = random.randint(0,(len(self.chunks) - int(5000 / self.min_interval)))          # Wrapping around to new random location
            return 0, True
            
    def get_random_action(self) -> int:
        """ Returns random number between [1,3], inclusive
        
        Arguments:
            None

        Return:
            rand_int (int): Random int between 1 and 3
        
        Side Effects:
            None
        """
        return random.randint(1,3)

    def begin(self) -> list:
        """ Increments the current chunk by one and returns the first chunk
        
        Arguments:
            None

        Return:
            first_chunk (list): The first chunk of the list
        
        Side Effects:
            None
        """

        self.curr_chunk += 1
        return self.chunks[0]




