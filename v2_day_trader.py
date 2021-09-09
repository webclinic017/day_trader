import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
import alpaca_trade_api as tradeapi
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pickle
import os.path
import os
import pandas_ta as pta
import threading
import time
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
import shutil
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
from dateutil import parser
from datetime import datetime



PAUSE_SIG = False

class Enviroment:
    
    chunks: list
    curr_chunk: int
    buy_prices: list
    curr_money: int
    closing_prices: list

    def __init__(self, chunks, closing_prices):
        
        self.chunks = chunks
        self.curr_chunk = 0
        self.buy_prices = []
        self.curr_money = 1000
        self.prev_money = 1000
        self.closing_prices = closing_prices
    
    def reset(self):
        self.curr_chunk = random.randint(0,(len(self.chunks) - 1))
        self.buy_prices = []
        self.curr_money = 1000

    def get_current_money(self):
        
        cash = self.curr_money
        hypothetical = 0

        for buy_price in self.buy_prices:
            hypothetical += (100 * (self.closing_prices[self.curr_chunk] / buy_price))
        
        return round((cash+hypothetical), 2)


    # Could also end it if we didn't get over 10% return in a day
    def get_reward(self, current, decay):
        
        reward = 0

        if self.buy_prices:
            
            for buy_price in self.buy_prices:
                reward += current - buy_price
                self.curr_money += (100 * (current / buy_price))
            
            self.curr_money = round(self.curr_money, 1)
            return round(reward, 1)

        else:
            return 0

    def get_current_state(self):
        
        curr_state = np.copy(self.chunks[self.curr_chunk])
        toal_holdings = len(self.buy_prices)
        curr_state = np.append(curr_state, toal_holdings)
        curr_state = np.append(curr_state, self.curr_money)

        return curr_state

    def test_step(self, action, prev_chunk, decay):
       
        if self.curr_chunk + 1 < int(len(self.chunks)):
 
            last_close = 19
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.curr_money > 100:
                    self.buy_prices.append(prev_chunk[last_close])       # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 100

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                print(f"     Decided to buy stock at price: {prev_chunk[last_close]} || {self.get_current_money()} || {self.curr_chunk} \n")
                

            elif action == 2:   # Holding the stock
                reward = 0      # Do nothing - no reward
                print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(prev_chunk[last_close], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_prices = []
                print(f"     Decided to sell stock at price: {prev_chunk[last_close]} || {self.get_current_money()} || {self.curr_chunk} \n")

            if self.curr_chunk > 2999 and self.curr_chunk % 3000 == 0:        # Checking if we made 4% the past month

                if self.get_current_money() > int(self.prev_money*1.02):
                    return 10 * reward, False

            return reward,  False
        
        else:
            return 0, True

    def step(self, action, prev_chunk, decay):

        if self.curr_chunk + 1 < int(len(self.chunks)):
 
            last_close = 9
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.curr_money > 100:
                    self.buy_prices.append(self.closing_prices[self.curr_chunk -1])       # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 100

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {prev_chunk[last_close]} || {self.get_current_money()} || {self.curr_chunk} \n")
                

            elif action == 2:   # Holding the stock
                reward = 0      # Do nothing - no reward
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_prices = []
                #print(f"     Decided to sell stock at price: {prev_chunk[last_close]} || {self.get_current_money()} || {self.curr_chunk} \n")

            
            if self.curr_chunk > 1654 and self.curr_chunk % 1655 == 0:        # Checking if we made 4% the past month

                if self.get_current_money() < int(self.prev_money*1.01):
                    return 0, True

                else:
                    self.prev_money = self.get_current_money()
                    return reward,  False

            return reward,  False
        
        else:
            self.curr_chunk = random.randint(0,(len(self.chunks) - 1))          # Wrapping around to new random location
            return self.step(action, prev_chunk, decay)
            

# Need to clean the data, restrict the hours were calling from the api so each day has a uniform number of hours, can't trade afterhours

    def get_random_action(self):
        return random.randint(1,3)

    def begin(self):
        self.curr_chunk += 1
        return self.chunks[0]

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def create_rsi(closing_price:list):

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

# RSI is somehow making it shit?

def fill_df(df: pd.DataFrame(), ticker: str) -> list:

    times = list(df.index)
    new_times = []

    if times:
        current_closing_prices = df[(ticker, 'close')]
        filled_closing_prices = []

        starting_time = str(times[0]).split(" ")[1]

        if starting_time != "10:00:00-04:00":
            
            curr_last_min = parser.parse(str(times[0]).split(" ")[1])
            actual_last_min = parser.parse("10:00:00-04:00")

            diff = int((curr_last_min - actual_last_min).seconds/60)

            if diff != 0:
                for k in range(diff):
                    new_times.append(times[0] + timedelta(minutes=k+1))
                    filled_closing_prices.append(current_closing_prices[0])


        for i in range(len(times)):

            filled_closing_prices.append(current_closing_prices[i])
            new_times.append(times[i])

            if i+1 < len(times):
                
                diff = int((times[i+1] - times[i]).seconds / 60)

                if diff != 1:
                    for j in range(diff - 1):
                        filled_closing_prices.append(current_closing_prices[i])
                        new_times.append(times[i] + timedelta(minutes=j+1))
            
            else:
                
                curr_last_min = parser.parse(str(times[i]).split(" ")[1])
                actual_last_min = parser.parse("15:30:00-04:00")

                diff = int((actual_last_min - curr_last_min).seconds/60)

                if diff != 0:
                    for k in range(diff):
                        new_times.append(times[i] + timedelta(minutes=k+1))
                        filled_closing_prices.append(current_closing_prices[i])

        
        if len(filled_closing_prices) != 331:
            for t in new_times:
                print(t)
        return filled_closing_prices
    
    else: 
        return []


def create_testing_chunks():
    
    if os.path.isfile("cache/test_data.pkl"):
        with open("cache/test_data.pkl", "rb") as f:
            chunks_by_days =  pickle.load(f)
    
    else:

        api = tradeapi.REST()
        start_date = date(2021, 6, 7)
        end_date = date(2021, 9, 7)

        dates = []
        for single_date in daterange(start_date, end_date):
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
    chunks = get_emas(chunks_by_days, 10, rsi)

    return chunks


def create_training_chunks():
    
    if os.path.isfile("chunks_by_days_SPY.pkl"):
        with open("chunks_by_days_SPY.pkl", "rb") as f:
            chunks_by_days_SPY =  pickle.load(f)
        with open("chunks_by_days_VTI.pkl", "rb") as f:
            chunks_by_days_VTI =  pickle.load(f)
        with open("chunks_by_days_VXUS.pkl", "rb") as f:
            chunks_by_days_VXUS =  pickle.load(f)
        with open("chunks_by_days_BND.pkl", "rb") as f:
            chunks_by_days_BND =  pickle.load(f)


    else:

        api = tradeapi.REST()
        start_date = date(2018, 8, 26)
        end_date = date(2021, 8, 26)

        dates = []
        for single_date in daterange(start_date, end_date):
            dates.append(str(single_date))

        chunks = []
        chunks_by_days_SPY = {}
        chunks_by_days_VTI = {}
        chunks_by_days_VXUS = {}
        chunks_by_days_BND = {}
        
        skipped_dates = []
        for i in tqdm(range(int(len(dates)))):
            
            SPY_barset = api.get_barset(symbols=['SPY'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
            SPY_barset = fill_df(SPY_barset, 'SPY')

            VTI_barset = api.get_barset(symbols=['VTI'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 39
            VTI_barset = fill_df(VTI_barset, 'VTI')

            VXUS_barset = api.get_barset(symbols=['VXUS'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
            VXUS_barset = fill_df(VXUS_barset, 'VXUS')

            BND_barset = api.get_barset(symbols=['BND'], timeframe='1Min', limit=1000, start=dates[i]+'T10:00:00-04:00' , end=dates[i]+'T15:30:00-04:00').df # Getting all minute information for each day in the past year, whis is it length of 1000??? Should be 390
            BND_barset = fill_df(BND_barset, 'BND')
       
            
            if len(SPY_barset) == 0:
                skipped_dates.append(dates[i])
                continue
            
            for k in range(331):

                if k+10 < 331:              # Creating segments of 10 blocks, moving by one each time

                    if dates[i-1] in chunks_by_days_SPY:
                        chunks_by_days_SPY[dates[i-1]].append(SPY_barset[k:k+10])
                    else:
                        chunks_by_days_SPY[dates[i-1]] = [SPY_barset[k:k+10]]
                    
                    if dates[i-1] in chunks_by_days_VTI:
                        chunks_by_days_VTI[dates[i-1]].append(VTI_barset[k:k+10])
                    else:
                        chunks_by_days_VTI[dates[i-1]] = [VTI_barset[k:k+10]]
                    
                    if dates[i-1] in chunks_by_days_VXUS:
                        chunks_by_days_VXUS[dates[i-1]].append(VXUS_barset[k:k+10])
                    else:
                        chunks_by_days_VXUS[dates[i-1]] = [VXUS_barset[k:k+10]]
                    
                    if dates[i-1] in chunks_by_days_BND:
                        chunks_by_days_BND[dates[i-1]].append(BND_barset[k:k+10])
                    else:
                        chunks_by_days_BND[dates[i-1]] = [BND_barset[k:k+10]]

        
        with open("chunks_by_days_SPY.pkl", 'wb') as f:
            pickle.dump(chunks_by_days_SPY, f)
        with open("chunks_by_days_VTI.pkl", 'wb') as f:
            pickle.dump(chunks_by_days_VTI, f)
        with open("chunks_by_days_VXUS.pkl", 'wb') as f:
            pickle.dump(chunks_by_days_VXUS, f)
        with open("chunks_by_days_BND.pkl", 'wb') as f:
            pickle.dump(chunks_by_days_BND, f)

    new_chunks_by_days_SPY, closing_prices = transform_relative(chunks_by_days_SPY, True)
    new_chunks_by_days_VTI = transform_relative(chunks_by_days_VTI)
    new_chunks_by_days_VXUS = transform_relative(chunks_by_days_VXUS)
    new_chunks_by_days_BND = transform_relative(chunks_by_days_BND)

    SPY_chunks = get_emas(chunks_by_days_SPY, new_chunks_by_days_SPY, 10)
    VTI_chunks = get_emas(chunks_by_days_VTI, new_chunks_by_days_VTI, 10)
    VXUS_chunks = get_emas(chunks_by_days_VXUS, new_chunks_by_days_VXUS, 10)
    BND_chunks = get_emas(chunks_by_days_BND, new_chunks_by_days_BND, 10)

    chunks = [SPY_chunks, VTI_chunks, VXUS_chunks, BND_chunks]

    total_chunks = combine_chunks(chunks)

    return total_chunks, closing_prices

            
def combine_chunks(chunks: list):

    SPY_chunks = chunks[0]
    chunks.pop(0)
    total_chunks = []

    for i in range(len(SPY_chunks)):
        
        curr_chunk = SPY_chunks[i]

        for j in range(len(chunks)):
            curr_chunk += chunks[j][i]
        
        total_chunks.append(curr_chunk)
    
    return total_chunks


def transform_relative(chunks_by_days, spy: bool = False):

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

            if spy and i-1 > 48:            
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

def get_closing_daily_price(chunks_by_days: dict):

    closing_prices = []

    for day in chunks_by_days:
        
        closing_day = []
        chunks = chunks_by_days[day]

        for chunk in chunks:

            for i in range(len(chunk)):
                closing_day.append(chunk[i])
        
        closing_prices.append(round((sum(closing_day)/len(closing_day)),2))
    
    return closing_prices

def create_ema(day_average_list: list, num_days: int):
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

def get_emas(chunks_by_days: dict, new_chunks_by_days: dict, num_of_min):

   # Theres a big chunk that should go here, I think making daily average list?
    day_average_list = []

    for day in chunks_by_days:

        chunks = chunks_by_days[day]
        day_total = 0
        day_avg = 0

        for chunk in chunks:
            for i in range(len(chunk)):
                day_total += chunk[i]
        
        day_avg = day_total / (len(chunks) * num_of_min)
        day_average_list.append(day_avg)

    
    ema_daily_50 = create_ema(day_average_list, 50)
    ema_daily_20 = create_ema(day_average_list, 20)
    ema_daily_10 = create_ema(day_average_list, 10)
    ema_daily_5 = create_ema(day_average_list, 5)
    
    
    total_chunks = []
    ema_ranges = {49:ema_daily_50, 19:ema_daily_20, 9:ema_daily_10, 4:ema_daily_5}
   
        
    i = 0
    for k in range(49, len(new_chunks_by_days.keys())):                         # Starting at 49 because all other ranges fall under this, this is upper limit
        
        for chunk in new_chunks_by_days[list(new_chunks_by_days.keys())[k]]: 
            
            chunk.append(ema_ranges[49][i])
            chunk.append(ema_ranges[19][i])
            chunk.append(ema_ranges[9][i])
            chunk.append(ema_ranges[4][i])

            total_chunks.append(chunk)

        i += 1
    
    return total_chunks

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]      # Is this the q value then?
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.     # So q value for all possible actions, highest is chosen
    """
    learning_rate = 0.001       #Exploration rate
    init = tf.keras.initializers.HeUniform()        #Certain normalizer?
    model = keras.Sequential()      #Must be type of neural net?
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))      #Maybe this is copying over the weights
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7         # Learning rate
    discount_factor = 0.618     # Not sure? 

    MIN_REPLAY_SIZE = 1000      
    if len(replay_memory) < MIN_REPLAY_SIZE:        # Only do this function when we've gone through atleast 1000 steps?
        return

    batch_size = 64 * 2     # Getting random 128 batch sample from 
    mini_batch = random.sample(replay_memory, batch_size)       # Grabbing said random sample
    current_states = np.array([transition[0] for transition in mini_batch])     # Getting all the states from your sampled mini batch, because index 0 is the observation
    current_qs_list = model.predict(current_states)     # Predict the q values based on all the historical state
    new_current_states = np.array([transition[3] for transition in mini_batch]) # Getting all of the states after we executed our action? 
    future_qs_list = target_model.predict(new_current_states)       # the q values resulting in our action

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):       # Looping through our randomly sampled batch
        if not done:                                                                                # If we havent finished the game or died?
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])                 # Calculuting max value for each step using the discount factor
        else:
            max_future_q = reward                                                                   # if we finished then max is just the given reqard
        
        action -= 1
        current_qs = current_qs_list[index]     # Getting current value of q's
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q        # Updating the q values

        X.append(observation)           # Creating model input based off this
        Y.append(current_qs)            #
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)             # Fitting the model to the new input

def save_state(model: object, target_model: object, it_num: int, replay_mem: deque, X: list, Y: list):
    
    model.save(f"model_{it_num}")

    target_model.save(f"target_model_{it_num}")

    with open(f"replay_mem_{it_num}.pkl", 'wb') as f:
        pickle.dump(replay_mem, f)
    
    with open(f"X_{it_num}.pkl", 'wb') as f:
        pickle.dump(X, f)
    
    with open(f"Y_{it_num}.pkl", 'wb') as f:
        pickle.dump(Y, f)
    
    if it_num != 0:
        shutil.rmtree(f"model_{it_num-1}")
        shutil.rmtree(f"target_model_{it_num-1}")
        os.remove(f"replay_mem_{it_num-1}.pkl")
        os.remove(f"X_{it_num-1}.pkl")
        os.remove(f"Y_{it_num-1}.pkl")
    
def simulate(env: Enviroment):

    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
    max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
    decay = 0.01       # rate of increasing exploitation vs exploration - Change decay rate, we have 30,000 examples but reach full optimization after 1000
    episode = 0
    total_segment_reward = 0

    X = []
    y = []

    model = agent((58,), 3)
    target_model = agent((58,), 3) # Making neural net with input layer equal to state space size, and output layer equal to action space size
    target_model.set_weights(model.get_weights())
    
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
                predicted = model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                action = np.argmax(predicted) 
            
            reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
            new_state = env.get_current_state()                 # Getting the next step
            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                train(env, replay_memory, model, target_model, done)            # training the main model
                
        
        episode += 1
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        target_model.set_weights(model.get_weights())

        print(f"Made it to ${env.get_current_money()} -  it number: {i} - epsilon: {epsilon}")
        X.append(len(X) + 1)
        y.append(env.get_current_money())
        total_segment_reward = 0

        save_state(model, target_model, (episode-1), replay_memory, X, y)
        

    X, Y = np.array(X).reshape(-1,1), np.array(y).reshape(-1,1)
    plt.plot(X, y)  # Plot the chart
    plt.xlabel("Iteration Number")
    plt.ylabel("Number of decisions made")
    plt.show()
    plt.plot( X, LinearRegression().fit(X, Y).predict(X))
    plt.show()


    # Give reward after each trade vs every 100 try that

    with open("old_x.pkl", 'wb') as f:
        pickle.dump(X, f)

    with open("old_y.pkl", 'wb') as f:
        pickle.dump(y, f)

def test(env):

    epsilon = 0.05 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
    max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
    decay = 0.01       # rate of increasing exploitation vs exploration - Change decay rate, we have 30,000 examples but reach full optimization after 1000
    episode = 0

    done = False

    model = keras.models.load_model(f"cache/model_{431}")
    target_model = keras.models.load_model(f"cache/target_model_{431}") # Making neural net with input layer equal to state space size, and output layer equal to action space size
    with open(f"cache/replay_mem_{431}.pkl", 'rb') as f:
        replay_memory = pickle.load(f)
    
    i = 0
    while(not done):
        
        done = False
        steps_to_update_target_model = 0
        env.reset()
        
        steps_to_update_target_model += 1 
        random_number = np.random.rand()
        current_state = env.get_current_state()
        

        if random_number <= epsilon:  # Explore  
            action = env.get_random_action() # Just randomly choosing an action
        
        else: #Exploitting
            current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
            predicted = model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
        
        reward, done = env.test_step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
        new_state = env.get_current_state()                 # Getting the next step
        replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

        # 3. Update the Main Network using the Bellman Equation
        if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
            train(env, replay_memory, model, target_model, done)            # training the main model
        
        if steps_to_update_target_model % 100 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            target_model.set_weights(model.get_weights())
        

        print(f"Current money =  ${env.get_current_money()} -  it number: {i} / 5817 - epsilon: {epsilon}")
        i += 1
    
def trend_analysis():

    with open("results/X(3)_301.pkl", 'rb') as f:
        X = np.array(pickle.load(f))
    
    with open("results/Y(3)_301.pkl", 'rb') as f:
        Y = np.array(pickle.load(f))

    plt.plot(X,Y,'o')
    coefs = poly.polyfit(X, Y, 10)
    ffit = poly.polyval(X, coefs)
    plt.plot(X, ffit)

    plt.show()





def main():

    chunks, closing_prices = create_training_chunks()
    env = Enviroment(chunks, closing_prices)
    simulate(env)

   #chunks = create_testing_chunks()
   #env = Enviroment(chunks)
   #test(env)

   #trend_analysis()



if __name__ == '__main__':
    main()