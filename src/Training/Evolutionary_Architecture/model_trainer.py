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
from model import Model
import time
import multiprocessing as mp
import model


def daterange(start_date: datetime, end_date: datetime) -> list:
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

def create_rsi(closing_price:list) -> list:
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

def fill_df(frame: object, ticker: str):
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

def create_training_chunks(minute_interval: int):
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
        for single_date in daterange(start_date, end_date):
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
            SPY_barset, SPY_bar_vol = fill_df(SPY_barset, 'SPY')  

            VTI_barset = api.get_barset(symbols=['VTI'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 39
            VTI_barset, VTI_bar_vol = fill_df(VTI_barset, 'VTI')

            VXUS_barset = api.get_barset(symbols=['VXUS'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
            VXUS_barset, VXUS_bar_vol = fill_df(VXUS_barset, 'VXUS')

            BND_barset = api.get_barset(symbols=['BND'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
            BND_barset, BND_bar_vol = fill_df(BND_barset, 'BND')

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

    SPY_prices, SPY_closing_prices = make_stationary(SPY_prices, minute_interval, True)
    VTI_prices = make_stationary(VTI_prices, minute_interval)
    VXUS_prices = make_stationary(VXUS_prices, minute_interval)
    BND_prices = make_stationary(BND_prices, minute_interval)

    SPY_vol = make_stationary(SPY_vol, minute_interval)
    VTI_vol = make_stationary(VTI_vol, minute_interval)
    VXUS_vol = make_stationary(VXUS_vol, minute_interval)
    BND_vol = make_stationary(BND_vol, minute_interval)

    price_chunks = [VTI_prices, VXUS_prices, BND_prices]
    vol_chunks = [VTI_vol, VXUS_vol, BND_vol]

    total_chunks = combine_chunks(price_chunks, vol_chunks, SPY_prices, SPY_vol)

    return total_chunks, SPY_closing_prices
    
def segment_chunks(chunks_by_days: dict) -> list:
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

def get_closing_prices(chunks_by_days: dict) -> list:
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

def segment_and_avg(list_to_seg: list, interval: int):

    segmented = [list_to_seg[i:i + interval] for i in range(0, len(list_to_seg), interval)]
    avg_list = []

    for sub_list in segmented:
        avg = round(sum(sub_list) / len(sub_list),3)
        avg_list.append(avg)
    
    return avg_list

def make_stationary(prices: dict, min_interval: int, get_close: bool = False):
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

    new_prices = segment_and_avg(new_prices, min_interval)
    old_prices = segment_and_avg(old_prices, min_interval)

    chunks = []
    for i in range(len(new_prices)):
        
        if i+10 < len(new_prices):
            chunks.append(new_prices[i:i+10])
            closing_prices.append(old_prices[i:i+10][-1])
    
    if get_close:
        return chunks, closing_prices
    else:
        return chunks

def combine_chunks(price_chunks: list, vol_chunks: list, price_SPY_chunks: list, vol_SPY_chunks: list) -> list:
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

def transform_relative(chunks_by_days, spy: bool = False) -> dict:
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

def get_closing_daily_price(chunks_by_days: dict) -> list:
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

def save_state( model: object, target_model: object, it_num: int, replay_mem: deque, X: list, Y: list, max_profits: list, ver: int) -> None:
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
    
def simulate(decay: float, ver: int, training_iterations: int, model_config: list) -> None:
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

    # Instead, write all the data out to a file, Get the 8 most recently wrote to files and use that I think

    if not os.path.exists(f'models/{ver}'):
        os.makedirs(f'models/{ver}')

    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
    max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
    episode = 0
    total_segment_reward = 0

    X = []
    y = []
    monies = []
    max_profits = []

    models = model.Model((model_config[0], model_config[1]))

    models.target_model.set_weights(models.model.get_weights())
    
    replay_memory = deque(maxlen=100_000)
    chunks, closing_prices = create_training_chunks(60) 
    env = TrainingEnviroment(chunks, closing_prices, 60)

    with open(f"models/{ver}/architecture.pkl", 'wb') as f:
        pickle.dump(model_config, f)

    for i in tqdm(range(training_iterations)):
        
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

                current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])

                predicted = list(models.model.predict(current_reshaped).flatten())          # Predicting best action, not sure why flatten (pushing 2d into 1d)
                action = np.argmax(predicted) 
                action += 1
            
            reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
            
            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()
            
            index = -4
            for k in range(9,-1,-1):

                try:
                    new_state[index] = np.log(env.buy_prices[k])
                except:
                    new_state[index] = 0
                index -= 1

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

            # 3. Update the Main Network using the Bellman Equation, can maybe do this for every cpu we have and paralize the training process
            if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                models.train(replay_memory, done)            # training the main model
                    
            monies.append(env.get_current_money())
            with open(f"models/{ver}/monies.pkl", 'wb') as f:
                pickle.dump(monies, f)
        
        monies = []
        with open(f"models/{ver}/monies.pkl", 'wb') as f:
            pickle.dump(monies, f)
        time.sleep(3)
            


        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        models.target_model.set_weights(models.model.get_weights())

        print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} -  it number: {i} - epsilon: {epsilon}")
        X.append(len(X) + 1)
        max_profits.append(env.max_profit)
        y.append(env.get_current_money())
        total_segment_reward = 0

        save_state(models.model, models.target_model, (episode-1), replay_memory, X, y, max_profits, ver)
    
    monies = [-1]
    with open(f"models/{ver}/monies.pkl", 'wb') as f:
        pickle.dump(monies, f)
    time.sleep(3)
              
def train_from_save(env: TrainingEnviroment, iteration: int, model_name: str, target_model_name: str, replay_mem_name: str, epsilon: float, decay: int, ver: int, max_it: int):
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

    models.model = keras.models.load_model(model_name)
    models.target_model = keras.models.load_model(target_model_name)
    with open(replay_mem_name, 'rb') as f:
        replay_memory = pickle.load(f)



    for i in tqdm(range(max_it - iteration)):

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
                
                current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
                predicted = models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                action = np.argmax(predicted) 
                action += 1
                

            reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
            
            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()

            index = -4
            for k in range(9,-1,-1):

                try:
                    new_state[index] = np.log(env.buy_prices[k])
                except:
                    new_state[index] = 0
                index -= 1

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                models.train(replay_memory, done)            # training the main model
                
        
        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        models.target_model.set_weights(models.model.get_weights())

        print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} -  it number: {i} - epsilon: {epsilon}")
        X.append(len(X) + 1)
        max_profits.append(env.max_profit)
        y.append(env.get_current_money())
        total_segment_reward = 0

        save_state(models.model, models.target_model, episode, replay_memory, X, y, max_profits, ver)

def test_v2(version: str, shared_dict: dict) -> None:
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

    chunks, closing_prices = create_training_chunks(60) 
    env = TrainingEnviroment(chunks, closing_prices, 60)

    models = Model()

    models.model = keras.models.load_model(f"models/{version}/model_{version}")
    models.target_model = keras.models.load_model(f"models/{version}/model_{version}") # Making neural net with input layer equal to state space size, and output layer equal to action space size
    with open(f"models/{version}/model_{version}", 'rb') as f:
        replay_memory = pickle.load(f)
    
    money_made = 0
    

    for i in range(3):
        
        overal_done = False
        env.load_year_test()

        while(not overal_done):
            
            done = False
            current_state = env.get_current_state()

            current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
            predicted = models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
            action += 1
            
                
            reward, weekly_done, overal_done = env.test_step_v2(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state

            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.curr_money
            
            index = -4
            for k in range(9,-1,-1):

                try:
                    new_state[index] = np.log(env.buy_prices[k])
                except:
                    new_state[index] = 0
                index -= 1

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory


            if weekly_done:
                print(f"We didn't meet goal profit, made it to: {env.get_current_money()*1} , should have made it to: {env.goal_profit*1}")
                money_made += (env.get_current_money()*1) - 1000
                print(f"Current made money: {money_made}")

                env.goal_profit = 1000 * env.weekly_return()
                env.curr_money = 1000
                env.buy_prices = []
                env.prev_money = 1000
                env.max_profit = 1000
                env.num_chunks = 0


    shared_dict[version].append(int(money_made/3))
    
def test(version: str, shared_dict: dict) -> None:
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

    chunks, closing_prices = create_training_chunks(60) 
    env = TrainingEnviroment(chunks, closing_prices, 60)


    models = Model()
    total_money = 0

    models.model = keras.models.load_model(f"models/{version}/model_{version}")
    models.target_model = keras.models.load_model(f"models/{version}/target_model_{version}") # Making neural net with input layer equal to state space size, and output layer equal to action space size
    with open(f"models/{version}/replay_mem_{version}", 'rb') as f:
        replay_memory = pickle.load(f)
  

    for i in range(3):
        
        done = False
        env.load_year_test()


        while(not done):
            
            done = False
            steps_to_update_target_model = 0
            
            steps_to_update_target_model += 1 
            random_number = np.random.rand()
            current_state = env.get_current_state()

            current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
            predicted = models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
            action += 1
                
            reward, done = env.test_step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state

            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()
            
            index = -4
            for k in range(9,-1,-1):

                try:
                    new_state[index] = np.log(env.buy_prices[k])
                except:
                    new_state[index] = 0
                index -= 1

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

        total_money += env.get_current_money()
    
    shared_dict[version].append(int(total_money/3))

def trend_analysis(ver: str, it: str) -> None:
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
    
def test_trend_analysis(ver: str) -> None:
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

        print(Y)

       #plt.scatter(buy_pos, buy_Y, marker='v', color='g')
        #plt.scatter(sell_pos, sell_Y, marker='v', color='r')

        plt.plot(X, Y)
        m, b = np.polyfit(X, Y, 1)
        
        plt.plot(X, m*X+b)
        print(f"Slope: {m}")

        plt.plot(X, Y_base)
        m_b, b_b = np.polyfit(X, Y_base, 1)
        
        plt.plot(X, m_b*X+b_b)
        print(f"Base Slop: {m_b}")

        plt.show()

