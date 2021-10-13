import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
import alpaca_trade_api as tradeapi
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import pickle
import os.path
import os
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import shutil
import matplotlib.pyplot as plt
from dateutil import parser
from training_enviroment import TrainingEnviroment
import sys
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from model import Model


class ModelTrainer():

    models: Model

    def daterange(self, start_date: datetime, end_date: datetime) -> list:
        """ Outputs a list of dates from the given start
        to the given end

        Arguments:
            start_date (datetime): The date to start at

            end_date (datetime): The date to end at

        Return:
            dates (list): List of all dates inbetween the start and finish
        
        Side Effects:
            None
        """

        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def create_rsi(self, closing_price:list) -> list:
        """ Creates list of rising stock index in correlation to 
        a list of prices, not currently in use

        Arguments:
            closing_price (list): List of all prices to be converted

        Return:
            rsi (list): List of relative stock index in correlation with the prices
        
        Side Effects:
            None
        """

        df = pd.DataFrame()
        df["close"] = pd.Series(closing_price)

        close_delta = df['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        ma_up = up.ewm(com = 14 - 1, adjust=True, min_periods = 14).mean()
        ma_down = down.ewm(com = 14 - 1, adjust=True, min_periods = 14).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))

        rsi = np.array(rsi)
        nan =np.isnan(rsi)
        rsi[nan]=0.0

        return rsi

    def fill_df(self, frame: object, ticker: str):
        """ Finds missing minutes in pandas data frame index and
        approprialtey fills those missing minutes with the general average or, if available
        the price that came before it


        Arguments:
            df (DataFrame): The returned data frame from Alpaca's API

            ticket (str): The associated ticker with the data frame

        Return:
            filled_closing_prices (list): List of completed closing prices with no missing time indexes

            filled_volume (list): List of completed volume indicators, with no missing time indexes
        
        Side Effects:
            None
        """

        times = list(frame.index)
        new_times = []

        if times:
            
            current_closing_prices = frame[(ticker, 'close')]
            filled_closing_prices = []

            current_volume = frame[(ticker, 'volume')]
            filled_volume = []


            starting_time = str(times[0]).split(" ")[1]

            if starting_time != "10:00:00-04:00":
                
                curr_last_min = parser.parse(str(times[0]).split(" ")[1])
                actual_last_min = parser.parse("10:00:00-04:00")

                diff = int((curr_last_min - actual_last_min).seconds/60)

                if diff != 0:
                    for k in range(diff):
                        new_times.append(times[0] + timedelta(minutes=k+1))
                        filled_closing_prices.append(current_closing_prices[0])
                        filled_volume.append(current_volume[0])


            for i in range(len(times)):

                filled_closing_prices.append(current_closing_prices[i])
                filled_volume.append(current_volume[i])

                new_times.append(times[i])

                if i+1 < len(times):
                    
                    diff = int((times[i+1] - times[i]).seconds / 60)

                    if diff != 1:
                        for j in range(diff - 1):
                            
                            filled_closing_prices.append(current_closing_prices[i])
                            filled_volume.append(current_volume[i])
                            new_times.append(times[i] + timedelta(minutes=j+1))
                
                else:
                    
                    curr_last_min = parser.parse(str(times[i]).split(" ")[1])
                    actual_last_min = parser.parse("15:30:00-04:00")

                    diff = int((actual_last_min - curr_last_min).seconds/60)

                    if diff != 0:
                        for k in range(diff):
                            
                            new_times.append(times[i] + timedelta(minutes=k+1))
                            filled_closing_prices.append(current_closing_prices[i])
                            filled_volume.append(current_volume[i])


            not_0 = len(filled_closing_prices) - filled_closing_prices.count(0)
            avg = round(sum(filled_closing_prices) / not_0,22)

            vol_not_0 = len(filled_volume) - filled_volume.count(0)
            vol_avg = round(sum(filled_volume) / vol_not_0, 2)

            for k in range(len(filled_closing_prices)):
                if filled_closing_prices[k] == 0:
                    filled_closing_prices[k] = avg
                    filled_volume[k] = vol_avg

            return filled_closing_prices, filled_volume
        
        else: 
            return [], []

    def create_testing_chunks(self):
        """ If not already created, creates list of 3 year date range, downloades
        associated data with it from Alpaca API. Takes said data, segements it into 10min chunks 
        and applies the nessesary transformations on it.


        Arguments:
            None

        Return:
            total_chunks (list): List of total segmented chunks

            closing_prices (list): List of actual dollar values to be associated with the chunks
        
        Side Effects:
            None
        """

        if os.path.isfile("cache/test_data.pkl"):
            with open("cache/test_data.pkl", "rb") as f:
                chunks_by_days =  pickle.load(f)
        
        else:

            api = tradeapi.REST()
            start_date = date(2021, 6, 7)
            end_date = date(2021, 9, 7)

            dates = []
            for single_date in self.daterange(start_date, end_date):
                dates.append(str(single_date) + 'T00:00:00-00:00')

            chunks = []
            chunks_by_days = {}

            for i in tqdm(range(int(len(dates)))):
                
                if i!=0: 
                    barset = api.get_barset(symbols=['SPY'], timeframe='1Min', limit=1000, start=dates[i-1], end=dates[i]) # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
                    barset = barset["SPY"]
                    
                    for k in range(len(barset)):

                        if k+10 < len(barset):              # Creating segments of 10 blocks, moving by one each time

                            chunk = []
                            for price in barset[k:k+10]:
                                chunk.append(price.o)
                                chunk.append(price.c)
                            
                            chunks.append(chunk)

                            if dates[i-1] in chunks_by_days:
                                chunks_by_days[dates[i-1]].append(chunk)
                            else:
                                chunks_by_days[dates[i-1]] = [chunk]
            
            with open("cache/test_data.pkl", 'wb') as f:
                pickle.dump(chunks_by_days, f)


        for day in chunks_by_days:
            chunks = chunks_by_days[day]

            for chunk in chunks:
                print(chunk)

        #closing_prices = get_closing_daily_price(chunks_by_days)
        #rsi = create_rsi(closing_prices)
        rsi = []
        chunks = self.get_emas(chunks_by_days, 10, rsi)

        return chunks

    def create_training_chunks(self, minute_interval: int):
        """ If not already created, creates list of 3 year date range, downloades
        associated data with it from Alpaca API. Takes said data, segements it into 10min chunks 
        and applies the nessesary transformations on it.


        Arguments:
            None

        Return:
            total_chunks (list): List of total segmented chunks

            closing_prices (list): List of actual dollar values to be associated with the chunks
        
        Side Effects:
            None
        """
        # Increased time intervals are just the average
        if os.path.isfile("cache/SPY_prices.pkl"):
            
            with open("cache/SPY_prices.pkl", "rb") as f:
                SPY_prices =  pickle.load(f)
            with open("cache/VTI_prices.pkl", "rb") as f:
                VTI_prices =  pickle.load(f)
            with open("cache/VXUS_prices.pkl", "rb") as f:
                VXUS_prices =  pickle.load(f)
            with open("cache/BND_prices.pkl", "rb") as f:
                BND_prices =  pickle.load(f)
            
            with open("cache/SPY_vol.pkl", "rb") as f:
                SPY_vol =  pickle.load(f)
            with open("cache/VTI_vol.pkl", "rb") as f:
                VTI_vol =  pickle.load(f)
            with open("cache/VXUS_vol.pkl", "rb") as f:
                VXUS_vol =  pickle.load(f)
            with open("cache/BND_vol.pkl", "rb") as f:
                BND_vol =  pickle.load(f)


        else:

            api = tradeapi.REST()
            start_date = date(2018, 8, 26)
            end_date = date(2021, 8, 26)

            dates = []
            for single_date in self.daterange(start_date, end_date):
                dates.append(str(single_date))

            SPY_prices = []
            VTI_prices = []
            VXUS_prices = []
            BND_prices = []

            SPY_vol = []
            VTI_vol = []
            VXUS_vol = []
            BND_vol = []
            
            skipped_dates = []
            for i in tqdm(range(int(len(dates)))):
                
                SPY_barset = api.get_barset(symbols=['SPY'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
                SPY_barset, SPY_bar_vol = self.fill_df(SPY_barset, 'SPY')  

                VTI_barset = api.get_barset(symbols=['VTI'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 39
                VTI_barset, VTI_bar_vol = self.fill_df(VTI_barset, 'VTI')

                VXUS_barset = api.get_barset(symbols=['VXUS'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
                VXUS_barset, VXUS_bar_vol = self.fill_df(VXUS_barset, 'VXUS')

                BND_barset = api.get_barset(symbols=['BND'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
                BND_barset, BND_bar_vol = self.fill_df(BND_barset, 'BND')

                base_len = len(SPY_barset)
                if base_len == 0 or len(VTI_barset) != base_len or len(VXUS_barset) != base_len or len(BND_barset) != base_len:
                    skipped_dates.append(dates[i])
                    continue
                
                else:
                    SPY_prices += SPY_barset
                    VTI_prices += VTI_barset
                    VXUS_prices += VXUS_barset
                    BND_prices += BND_barset

                    SPY_vol += SPY_bar_vol
                    VTI_vol += VTI_bar_vol
                    VXUS_vol += VXUS_bar_vol
                    BND_vol += BND_bar_vol
                
            with open("cache/SPY_prices.pkl", 'wb') as f:
                pickle.dump(SPY_prices, f)
            with open("cache/VTI_prices.pkl", 'wb') as f:
                pickle.dump(VTI_prices, f)
            with open("cache/VXUS_prices.pkl", 'wb') as f:
                pickle.dump(VXUS_prices, f)
            with open("cache/BND_prices.pkl", 'wb') as f:
                pickle.dump(BND_prices, f)

            with open("cache/SPY_vol.pkl", 'wb') as f:
                pickle.dump(SPY_vol, f)
            with open("cache/VTI_vol.pkl", 'wb') as f:
                pickle.dump(VTI_vol, f)
            with open("cache/VXUS_vol.pkl", 'wb') as f:
                pickle.dump(VXUS_vol, f)
            with open("cache/BND_vol.pkl", 'wb') as f:
                pickle.dump(BND_vol, f)


        SPY_prices, SPY_closing_prices = self.make_stationary(SPY_prices, minute_interval, True)
        VTI_prices = self.make_stationary(VTI_prices, minute_interval)
        VXUS_prices = self.make_stationary(VXUS_prices, minute_interval)
        BND_prices = self.make_stationary(BND_prices, minute_interval)

        SPY_vol = self.make_stationary(SPY_vol, minute_interval)
        VTI_vol = self.make_stationary(VTI_vol, minute_interval)
        VXUS_vol = self.make_stationary(VXUS_vol, minute_interval)
        BND_vol = self.make_stationary(BND_vol, minute_interval)

        price_chunks = [VTI_prices, VXUS_prices, BND_prices]
        vol_chunks = [VTI_vol, VXUS_vol, BND_vol]

        total_chunks = self.combine_chunks(price_chunks, vol_chunks, SPY_prices, SPY_vol)

        return total_chunks, SPY_closing_prices
        
    def segment_chunks(self, chunks_by_days: dict) -> list:
        """ Takes dictionary of chunks separated by the day and returns a list
        of the total chunks combined


        Arguments:
            chunks_by_days (dict): Dictionary of all of the chunks associated to their day

        Return:
            total_chunks (list): List of total segmented chunks

        Side Effects:
            None
        """

        total_chunks = []

        for day in chunks_by_days:

            for chunk in chunks_by_days[day]:
                total_chunks.append(chunk)
        
        return total_chunks

    def get_closing_prices(self, chunks_by_days: dict) -> list:
        """ Takes dictionary of chunks separated by the day and returns a list
        of the total closing prices


        Arguments:
            chunks_by_days (dict): Dictionary of all of the chunks associated to their day

        Return:
            closing_prices (list): List of total segmented closing prices

        Side Effects:
            None
        """

        closing_prices = []
        for day in chunks_by_days:
            chunks = chunks_by_days[day]

            for chunk in chunks:
                closing_prices.append(chunk[-1])
        
        return closing_prices

    def segment_and_avg(self, list_to_seg: list, interval: int):

        segmented = [list_to_seg[i:i + interval] for i in range(0, len(list_to_seg), interval)]
        avg_list = []

        for sub_list in segmented:
            avg = round(sum(sub_list) / len(sub_list),3)
            avg_list.append(avg)
        
        return avg_list

    def make_stationary(self, prices: dict, min_interval: int, get_close: bool = False):
        """ Takes list of prices and makes the data statioanry by applying
        log transformation to it. Captures orgiional list of prices if get_close
        is True

        Arguments:
            prices (list): List of all prices to transform

            get_close (bool): Boolean to dictate whether to save origional price numbers
            or not 

        Return:
            chunks (list): Total prices transformed and chunked into segments of 10

            closing_prices (list): List of closing prices 
        
        Side Effects:
            None
        """

        closing_prices = []
        price_df = pd.DataFrame()
        price_df['prices'] = prices
        price_df['new_prices'] = price_df['prices'].apply(np.log)

        new_prices = list(price_df['new_prices'])
        old_prices = list(price_df['prices'])

        new_prices = self.segment_and_avg(new_prices, min_interval)
        old_prices = self.segment_and_avg(old_prices, min_interval)

        chunks = []
        for i in range(len(new_prices)):
            
            if i+10 < len(new_prices):
                chunks.append(new_prices[i:i+10])
                closing_prices.append(old_prices[i:i+10][-1])
        
        if get_close:
            return chunks, closing_prices
        else:
            return chunks

    def combine_chunks(self, price_chunks: list, vol_chunks: list, price_SPY_chunks: list, vol_SPY_chunks: list) -> list:
        """ Takes list of all the different segmented chunks from the four stocks, in combination
        with their segements volume chunks and combines them all into one chunk for each iteration

        Arguments:
            price_chunks (list): List of lists for each stock and their prices

            vol_chunks (list): List of lists for each stock and their volume

            price_SPY_chunks (list): Only the prices for the SPY stock, so we can make sure its
            first

            vol_SPY_chunks (list): Only the volumes for the SPY stock, so we can make sure its
            first

        Return:
            total_chunks (list): List of all combined chunks
        
        Side Effects:
            None
        """

        total_chunks = []

        for i in range(len(price_SPY_chunks)):
            
            total_chunks.append([val for pair in zip(price_SPY_chunks[i], vol_SPY_chunks[i]) for val in pair])
            
            for k in range(len(price_chunks)):
                total_chunks[i] += [val for pair in zip(price_chunks[k][i], vol_chunks[k][i]) for val in pair]
        
        return total_chunks

    def transform_relative(self, chunks_by_days, spy: bool = False) -> dict:
        """ Takes dictionary of chunks associated with their days 
        and transforms the price to be a percentage relative to their
        neighbour

        Arguments:
            chunks_by_days (dict): Dictionary of the days to their chunks

            spy (bool): Boolean of whether current dictionary is the SPY stock
            or not, to dictate whether to capture the closing values or not.

        Return:
            new_chunks_by_days (dict): Dictionary of days associated with their
            relaitve prices
        
        Side Effects:
            None
        """

        prev_price = 0
        new_chunks_by_days = {}
        closing_prices = []
        
        i = 0
        for day in chunks_by_days:

            i += 1
            chunks = chunks_by_days[day]

            new_chunks = []

            for chunk in chunks:

                new_chunk = []   
                closing_prices.append(chunk[-1])

                for close_price in chunk:
                    
                    if prev_price != 0:

                        relative_price = 1 - (close_price/prev_price)
                        relative_price *= 100000
                        new_chunk.append(round(relative_price, 2))
                        prev_price = close_price
                    
                    else:
                        relative_price = 0
                        new_chunk.append(relative_price)
                        prev_price = close_price
                    
                
                new_chunks.append(new_chunk)
            
            new_chunks_by_days[day] = new_chunks
        if spy:
            return new_chunks_by_days, closing_prices

        else:
            return new_chunks_by_days                         

    def get_closing_daily_price(self, chunks_by_days: dict) -> list:
        """ Takes dictionary of chunks associated with their days 
        and gets the closing prices for each day on average

        Arguments:
            chunks_by_days (dict): Dictionary of the days to their chunks

        Return:
            closing_prices (list): List of closing day prices
        
        Side Effects:
            None
        """

        closing_prices = []

        for day in chunks_by_days:
            
            closing_day = []
            chunks = chunks_by_days[day]

            for chunk in chunks:

                for i in range(len(chunk)):
                    closing_day.append(chunk[i])
            
            closing_prices.append(round((sum(closing_day)/len(closing_day)),2))
        
        return closing_prices

    def create_ema(self, day_average_list: list, num_days: int) -> list:
        """ Takes the average price by day and computes the expoential moving average
        based on the time frame given

        Arguments:
            day_average_list (list): List of the average price per day

            num_days (int): The time frame to calculate the ema in 

        Return:
            ema_daily (list): List of ema for each given day
        
        Side Effects:
            None
        """
        
        ema_daily = []
        prev_ema = 0

        for i in range(len(day_average_list)):

            if i > (num_days - 2):                      # Start at 20 days

                current_20_days = day_average_list[i-(num_days-1):i]
                

                if prev_ema == 0:      # If this is the first set of 20
                    avg = sum(current_20_days) / (num_days)
                    ema = ((day_average_list[i] * (2/(i + 1))) + (avg * (1 - (2/(i+1)))))
                    prev_ema = avg
                    ema_daily.append(ema)

                else:
                    ema = ((day_average_list[i] * (2/(i + 1))) + (prev_ema * (1 - (2/(i+1)))))
                    prev_ema = ema
                    ema_daily.append(ema)
        
        return ema_daily

    def save_state(self, model: object, target_model: object, it_num: int, replay_mem: deque, X: list, Y: list, max_profits: list, ver: int) -> None:
        """ Takes current enviroemnt varaibles and saves them into pickled files or uses
        Keras's built in model save function to save map of varaibles for each iterations 
        
        Arguments:
            model (object): The current neural network

            target_model (object): The given nerual network to train

            it_num (int): The current iteration number

            replay_mem (deque): The current replay memory for iteration

            X (list): The list of iterations thus far

            Y (list): The list of end money values thus far

            max_profits (list): The list of the max profits achieved for each iteration
        
        Return:
            None

        Side Effects:
            Files created for all varaibles and files deleted for all varaibles
        """

        model.save(f"models/{ver}/model_{ver}_{it_num}")

        target_model.save(f"models/{ver}/target_model_{ver}_{it_num}")

        with open(f"models/{ver}/replay_mem_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(replay_mem, f)
        
        with open(f"models/{ver}/X_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(X, f)
        
        with open(f"models/{ver}/Y_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(Y, f)
        
        with open(f"models/{ver}/max_profits_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(max_profits, f)
        
        if it_num != 0:
            shutil.rmtree(f"models/{ver}/model_{ver}_{it_num-1}")
            shutil.rmtree(f"models/{ver}/target_model_{ver}_{it_num-1}")
            os.remove(f"models/{ver}/replay_mem_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/X_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/Y_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/max_profits_{ver}_{it_num-1}.pkl")
        
    def simulate(self, env: TrainingEnviroment, decay: float, ver: int) -> None:
        """ Overal model controller for the model and the training enviroments. 
        Controls the flow of inforamtion and helps simulate realtime data extraction
        for the model to learn on. Gives the model the current states, exectutes the action,
        updates the state and trains the model if nesseary. Also controls tracking of all
        relevant meta-data around training process.
        
        Arguments:
            env (TrainingEnviroment): The associated training enviroment with the model
        
        Return:
            None
        
        Side Effects:
            None
        """

        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        episode = 0
        total_segment_reward = 0

        X = []
        y = []
        max_profits = []

        self.models = Model(84, 3)
        self.models.target_model.set_weights(self.models.model.get_weights())
        
        replay_memory = deque(maxlen=100_000)


        for i in tqdm(range(1000)):
            
            
            done = False
            steps_to_update_target_model = 0
            env.reset()

            while(not done):

                total_segment_reward += 1
                steps_to_update_target_model += 1 
                random_number = np.random.rand()
                current_state = env.get_current_state()

                if random_number <= epsilon:  # Explore  
                    action = env.get_random_action() # Just randomly choosing an action
                
                else: #Exploitting

                    current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
                    predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                    action = np.argmax(predicted) 
                    action += 1
                
                reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
                
                new_state = current_state               
                new_state[-2] = env.num_chunks
                new_state[-3] = env.get_current_money()
                new_state[-4] = env.buy_price

                replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

                # 3. Update the Main Network using the Bellman Equation, can maybe do this for every cpu we have and paralize the training process
                if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                    self.models.train(replay_memory, done)            # training the main model
                    
            
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            self.models.target_model.set_weights(self.models.model.get_weights())

            print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} -  it number: {i} - epsilon: {epsilon}")
            X.append(len(X) + 1)
            max_profits.append(env.max_profit)
            y.append(env.get_current_money())
            total_segment_reward = 0

            self.save_state(self.models.model, self.models.target_model, (episode-1), replay_memory, X, y, max_profits, ver)
         
    def train_from_save(self, env: TrainingEnviroment, iteration: int, model_name: str, target_model_name: str, replay_mem_name: str, epsilon: float, decay: int, ver: int):
        """ Overal model controller for the model and the training enviroments. 
        Controls the flow of inforamtion and helps simulate realtime data extraction
        for the model to learn on. Gives the model the current states, exectutes the action,
        updates the state and trains the model if nesseary. Also controls tracking of all
        relevant meta-data around training process.
        
        Arguments:
            env (TrainingEnviroment): The associated training enviroment with the model

            iteration (int): Current iteration number

            model_name (str): Name of the model folder

            target_model_name (str): Name of the target model folder

            replay_mem_name (str): Name of the replay memory file

            epsilon (int): The current value of epsilon

            decay (int): The value of the decay rate
        
        Return:
            None
        
        Side Effects:
            None
        """

        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        episode = iteration
        total_segment_reward = 0

        X = []
        y = []
        max_profits = []

        self.models.model = keras.models.load_model(model_name)
        self.models.target_model = keras.models.load_model(target_model_name)
        with open(replay_mem_name, 'rb') as f:
            replay_memory = pickle.load(f)



        for i in tqdm(range(1000 - iteration)):

            done = False
            steps_to_update_target_model = 0
            env.reset()

            while(not done):

                total_segment_reward += 1
                steps_to_update_target_model += 1 
                random_number = np.random.rand()
                current_state = env.get_current_state()

                if random_number <= epsilon:  # Explore  
                    action = env.get_random_action() # Just randomly choosing an action
                
                else: #Exploitting
                    
                    current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
                    predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                    action = np.argmax(predicted) 
                    action += 1
                    

                reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
                
                new_state = current_state               
                new_state[-2] = env.num_chunks
                new_state[-3] = env.get_current_money()
                new_state[-4] = env.buy_price

                replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                    self.models.train(replay_memory, done)            # training the main model
                    
            
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            self.models.target_model.set_weights(self.models.model.get_weights())

            print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} -  it number: {i} - epsilon: {epsilon}")
            X.append(len(X) + 1)
            max_profits.append(env.max_profit)
            y.append(env.get_current_money())
            total_segment_reward = 0

            self.save_state(self.models.model, self.models.target_model, episode, replay_memory, X, y, max_profits, ver)

    def test_v2(self, env: TrainingEnviroment, model_name: str, target_model_name: str, replay_mem_name: str, ver: int) -> None:
        """ Overal model controller for the model and the training enviroments. 
        Controls the flow of inforamtion and helps simulate realtime data extraction
        for the model to learn on. Gives the model the current states, exectutes the action,
        updates the state and trains the model if nesseary. Also controls tracking of all
        relevant meta-data around training process. Only runs for the past year.
        
        Arguments:
            env (TrainingEnviroment): The associated training enviroment with the model

            model_name (str): Name of the model folder

            target_model_name (str): Name of the target model folder

            replay_mem_name (str): Name of replay memory file
        
        Return:
            None
        
        Side Effects:
            None
        """

        epsilon = 0.01 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time

        total_length = int(len(env.chunks)/3)
        overal_done = False
    
        current_money = []
        iteration = []
        action_list = []

        total_buy_positions = []
        total_sell_positions = []

        weekly_buy_positions = {}
        weekly_sell_positions = {}

        total_profit = []


        self.models.model = keras.models.load_model(model_name)
        self.models.target_model = keras.models.load_model(target_model_name) # Making neural net with input layer equal to state space size, and output layer equal to action space size
        with open(replay_mem_name, 'rb') as f:
            replay_memory = pickle.load(f)
        
        i = 0
        env.load_year_test()
        base_money = []
        buy_positions = []
        sell_positions = []
        base_price = env.get_current_stock_price()
        money_made = 0


        while(not overal_done):
            
            done = False
            steps_to_update_target_model = 0
            
            steps_to_update_target_model += 1 
            random_number = np.random.rand()
            current_state = env.get_current_state()
            base_money.append(1000 * (env.get_current_stock_price() / base_price))


            current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
            predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
            action += 1
                
            if action == 1 and env.curr_money > 999 and env.buy_price == -1:
                weekly_buy_positions[i] = list(predicted)
            
            elif action == 3 and env.buy_price > 0:
                weekly_sell_positions[i] = list(predicted)
                
            reward, weekly_done, overal_done = env.test_step_v2(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state

            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()
            new_state[-4] = env.buy_price

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory


            print(f"Current money =  ${env.get_current_money()} -  it number: {i} / {total_length} - epsilon: {epsilon}")
            current_money.append(env.get_current_money())
            iteration.append(i)
            i += 1
            

            if weekly_done:
                print(f"We didn't meet goal profit, made it to: {env.get_current_money()} , should have made it to: {env.goal_profit}")
                money_made += env.get_current_money() - 1000
                print(f"Current made money: {money_made}")
                
                total_buy_positions.append(weekly_buy_positions)
                total_sell_positions.append(weekly_sell_positions)

                weekly_buy_positions = {}
                weekly_sell_positions = {}

                total_profit.append(env.get_current_money() - 1000)
                
                env.goal_profit = 1010
                env.curr_money = 1000
                env.buy_price = -1
                env.prev_money = 1000
                env.max_profit = 1000
                env.num_chunks = 0


        
        with open(f"models/results/current_money_{ver}.pkl", 'wb') as f:
            pickle.dump(current_money, f)
        with open(f"models/results/total_buy_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(total_buy_positions, f)
        with open(f"models/results/total_sell_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(total_sell_positions, f)
        with open(f"models/results/total_profit_{ver}.pkl", 'wb') as f:
            pickle.dump(total_profit, f)
        with open(f"models/results/base_money_{ver}.pkl", 'wb') as f:
            pickle.dump(base_money, f)
        with open(f"models/results/iteration_{ver}.pkl", 'wb') as f:
            pickle.dump(iteration, f)
        with open(f"models/results/action_list_{ver}.pkl", 'wb') as f:
            pickle.dump(action_list, f)

        return money_made
        
    def test(self, env: TrainingEnviroment, model_name: str, target_model_name: str, replay_mem_name: str, ver: int) -> None:
        """ Overal model controller for the model and the training enviroments. 
        Controls the flow of inforamtion and helps simulate realtime data extraction
        for the model to learn on. Gives the model the current states, exectutes the action,
        updates the state and trains the model if nesseary. Also controls tracking of all
        relevant meta-data around training process. Only runs for the past year.
        
        Arguments:
            env (TrainingEnviroment): The associated training enviroment with the model

            model_name (str): Name of the model folder

            target_model_name (str): Name of the target model folder

            replay_mem_name (str): Name of replay memory file
        
        Return:
            None
        
        Side Effects:
            None
        """

        epsilon = 0.01 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time

        total_length = int(len(env.chunks)/3)
        done = False
    
        current_money = []
        iteration = []
        action_list = []

        self.models.model = keras.models.load_model(model_name)
        self.models.target_model = keras.models.load_model(target_model_name) # Making neural net with input layer equal to state space size, and output layer equal to action space size
        with open(replay_mem_name, 'rb') as f:
            replay_memory = pickle.load(f)
        
        i = 0
        env.load_year_test()
        base_money = []
        buy_positions = []
        sell_positions = []
        base_price = env.get_current_stock_price()


        while(not done):
            
            old_money = env.get_current_money()
            
            done = False
            steps_to_update_target_model = 0
            
            steps_to_update_target_model += 1 
            random_number = np.random.rand()
            current_state = env.get_current_state()
            base_money.append(1000 * (env.get_current_stock_price() / base_price))

            if random_number <= epsilon:  # Explore  
                action = env.get_random_action() # Just randomly choosing an action
            
            else: #Exploitting
                current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
                predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                action = np.argmax(predicted) 
                action += 1
                
            
            action_list.append(action)
            
            if action == 1:
                buy_positions.append(i)
            elif action == 3:
                sell_positions.append(i)
                
            reward, done = env.test_step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state

            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()
            new_state[-4] = env.buy_price
            
            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory


            print(f"Current money =  ${env.get_current_money()} -  it number: {i} / {total_length} - epsilon: {epsilon}")
            current_money.append(env.get_current_money())
            iteration.append(i)
            i += 1

            new_money = env.get_current_money()

        
        with open(f"models/results/current_money_{ver}.pkl", 'wb') as f:
            pickle.dump(current_money, f)
        with open(f"models/results/buy_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(buy_positions, f)
        with open(f"models/results/sell_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(sell_positions, f)
        with open(f"models/results/base_money_{ver}.pkl", 'wb') as f:
            pickle.dump(base_money, f)
        with open(f"models/results/iteration_{ver}.pkl", 'wb') as f:
            pickle.dump(iteration, f)
        with open(f"models/results/action_list_{ver}.pkl", 'wb') as f:
            pickle.dump(action_list, f)

        return env.get_current_money()

    def trend_analysis(self, ver: str, it: str) -> None:
        """ Takes the currnet data files for iterations and prices from
        hardcoded variables. Plots into line graph and plots best fit line 
        across for general trend analysis.
        
        Arguments:
            X_name (str): Name of X file

            Y_name (str): Name of Y file
        
        Return:
            None
        
        Side Effects:
            None
        """

        
        with open(f"models/{ver}/X_{ver}_{it}.pkl", 'rb') as f:
            X = list(pickle.load(f))
        with open(f"models/{ver}/Y_{ver}_{it}.pkl", 'rb') as f:
            Y = list(pickle.load(f))
        with open(f"models/{ver}/max_profits_{ver}_{it}.pkl", 'rb') as f:
            max_p = list(pickle.load(f))
        


        max_p = np.array(max_p)
        X = np.array(X)
        Y= np.array(Y)

    
        plt.plot(X, Y)
        m, b = np.polyfit(X, Y, 1)
        
        plt.plot(X, m*X+b)
        print(f"Slope: {m}")

        m_b, b_b = np.polyfit(X, max_p, 1)
        print(f"Max Profit Slop: {m_b}")

        plt.show()
        
    def test_trend_analysis(self, ver: str) -> None:
        """ Takes the currnet data files for iterations and prices from
        hardcoded variables. Plots into line graph and plots best fit line 
        across for general trend analysis.
        
        Arguments:
            X_name (str): Name of X file

            Y_name (str): Name of Y file
        
        Return:
            None
        
        Side Effects:
            None
        """

        
        with open(f"models/results/iteration_{ver}.pkl", 'rb') as f:
            X = list(pickle.load(f))
        with open(f"models/results/base_money_{ver}.pkl", 'rb') as f:
            Y_base = list(pickle.load(f))
        with open(f"models/results/current_money_{ver}.pkl", 'rb') as f:
            Y = list(pickle.load(f))
        with open(f"models/results/buy_pos_{ver}.pkl", 'rb') as f:
            buy_pos = list(pickle.load(f))
        with open(f"models/results/sell_pos_{ver}.pkl", 'rb') as f:
            sell_pos = list(pickle.load(f))

        buy_Y = []
        sell_Y = []

        for pos in buy_pos:
            buy_Y.append(Y_base[pos])

        for pos in sell_pos:
            sell_Y.append(Y_base[pos])


        Y_base = np.array(Y_base)
        X = np.array(X)
        Y= np.array(Y)

        plt.scatter(buy_pos, buy_Y, marker='v', color='g')
        plt.scatter(sell_pos, sell_Y, marker='v', color='r')

        plt.plot(X, Y)
        m, b = np.polyfit(X, Y, 1)
        
        plt.plot(X, m*X+b)
        print(f"Slop: {m}")

        plt.plot(X, Y_base)
        m_b, b_b = np.polyfit(X, Y_base, 1)
        
        plt.plot(X, m_b*X+b_b)
        print(f"Base Slop: {m_b}")

        plt.show()

    def buy_confidence_analysis(self, ver: str):
        
        
        with open(f"models/results/total_sell_pos_{ver}.pkl", 'rb') as f:
            total_sell = list(pickle.load(f))
        with open(f"models/results/total_buy_pos_{ver}.pkl", 'rb') as f:
            total_buy = list(pickle.load(f))
        with open(f"models/results/total_profit_{ver}.pkl", 'rb') as f:
            total_profit = list(pickle.load(f))

        buy_profit = []
        buy_loss = []

        X_input = []
        y_output = []
        clf = linear_model.SGDClassifier()


        for i in range(len(total_sell)):
            
            if total_profit[i] > 0:

                for pos in total_buy[i]:
                    buy_profit.append(total_buy[i][pos])
                    X_input.append(total_buy[i][pos])
                    y_output.append(1)

            else:

                for pos in total_buy[i]:
                    buy_loss.append(total_buy[i][pos])
                    X_input.append(total_buy[i][pos])
                    y_output.append(0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_input, y_output, test_size=0.2, random_state=1) 

        clf = clf.fit(X_train,y_train)

        y_pred = clf.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Accuracy:",metrics.confusion_matrix(y_test, y_pred))


def main():

    if not os.path.exists('cache'):
        os.makedirs('cache')

    if len(sys.argv) == 1:
        choice = int(input(" 1) Train from base level \n 2) Train from saved state \n 3) Test model \n 4) Weekly Test \n 5.) Trend analysis \n"))

        if choice == 1:

            decay = float(input("Please enter the desired decay value \n"))
            ver = input("Please enter the desired version  \n")
            minute_interval = int(input("Please enter the desired time interval in minutes \n"))

            if not os.path.exists(f'models/{ver}'):
                os.makedirs(f'models/{ver}')

            trainer_model = ModelTrainer()
            chunks, closing_prices = trainer_model.create_training_chunks(minute_interval)
            env = TrainingEnviroment(chunks, closing_prices, minute_interval)

            trainer_model.simulate(env, decay, ver)
        
        elif choice == 2:

            model_name = input("    Please enter the model name \n")
            target_model_name = input("    Please enter the target model name \n")
            replay_mem_name = input("    Please enter the replay memory name \n")
            iteration = int(input("    Please enter the iteration number \n"))
            epsilon = float(input("    Please enter the epsilon number \n"))
            decay = float(input("    Please enter the decay number \n"))
            ver = input("    Please enter the version number \n")
            minute_interval = int(input("Please enter the desired minute interval \n"))


            trainer_model = ModelTrainer()
            chunks, closing_prices = trainer_model.create_training_chunks(minute_interval)
            env = TrainingEnviroment(chunks, closing_prices, minute_interval)

            trainer_model.train_from_save(env, iteration, "models/" + model_name, "models/" + target_model_name, "models/" + replay_mem_name, epsilon, decay, ver)
        
        elif choice == 3:

            minute_interval = int(input("Please enter the desired minute interval \n"))
            ver = input("Please enter the model version \n")
            it = int(input("Please enter the model iteration \n"))

            trainer_model = ModelTrainer()
            chunks, closing_prices = trainer_model.create_training_chunks(minute_interval)

            results = 0
            for i in tqdm(range(minute_interval+1)):
                env = TrainingEnviroment(chunks, closing_prices, minute_interval)
                results += trainer_model.test(env, f"models/{ver}/model_{ver}_{it}", f"models/{ver}/target_model_{ver}_{it}", f"models/{ver}/replay_mem_{ver}_{it}.pkl", ver)

            print("average: " + str(results/(minute_interval+1)))
        
        elif choice == 4:

            minute_interval = int(input("Please enter the desired minute interval \n"))
            ver = input("Please enter the model version \n")
            it = int(input("Please enter the model iteration \n"))

            trainer_model = ModelTrainer()
            chunks, closing_prices = trainer_model.create_training_chunks(minute_interval)

            results = 0
            for i in tqdm(range(1)):
                env = TrainingEnviroment(chunks, closing_prices, minute_interval)
                results += trainer_model.test_v2(env, f"models/{ver}/model_{ver}_{it}", f"models/{ver}/target_model_{ver}_{it}", f"models/{ver}/replay_mem_{ver}_{it}.pkl", ver)

            print("average: " + str(results/(1)))

        elif choice == 5:

            ver = input("Please enter the version number: \n")
            it = input("Please enter the iteration number: \n")

            trainer_model = ModelTrainer()
            trainer_model.trend_analysis(ver, it)

    elif len(sys.argv) == 4:

        decay = float(sys.argv[1])
        ver = int(sys.argv[2])
        minute_interval = int(sys.argv[3])

        if not os.path.exists(f'cache/{ver}'):
            os.makedirs(f'cache/{ver}')

        trainer_model = ModelTrainer()
        chunks, closing_prices = trainer_model.create_training_chunks(minute_interval)
        env = TrainingEnviroment(chunks, closing_prices, minute_interval)

        trainer_model.simulate(env, decay, ver)



    

if __name__ == '__main__':
    main()