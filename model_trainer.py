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
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import shutil
import matplotlib.pyplot as plt
from dateutil import parser
from training_enviroment import TrainingEnviroment

# Should give the model the list of buy prices, not just the length, not sure if it has memeory or not 
class ModelTrainer():

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def create_rsi(self, closing_price:list):

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

    def fill_df(self, df: pd.DataFrame(), ticker: str) -> list:

        times = list(df.index)
        new_times = []

        if times:
            
            current_closing_prices = df[(ticker, 'close')]
            filled_closing_prices = []

            current_volume = df[(ticker, 'volume')]
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

    def create_training_chunks(self):
        
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

        print(len(SPY_prices))
        print(len(VTI_prices))
        print(len(VXUS_prices))
        print(len(BND_prices))

        print(len(SPY_vol))
        print(len(VTI_vol))
        print(len(VXUS_vol))
        print(len(BND_vol))

        SPY_prices, SPY_closing_prices = self.make_stationary(SPY_prices, True)
        VTI_prices = self.make_stationary(VTI_prices)
        VXUS_prices = self.make_stationary(VXUS_prices)
        BND_prices = self.make_stationary(BND_prices)

        SPY_vol = self.make_stationary(SPY_vol)
        VTI_vol = self.make_stationary(VTI_vol)
        VXUS_vol = self.make_stationary(VXUS_vol)
        BND_vol = self.make_stationary(BND_vol)

        price_chunks = [VTI_prices, VXUS_prices, BND_prices]
        vol_chunks = [VTI_vol, VXUS_vol, BND_vol]

        total_chunks = self.combine_chunks(price_chunks, vol_chunks, SPY_prices, SPY_vol)

        return total_chunks, SPY_closing_prices
        
    def segment_chunks(self, chunks_by_days: list):

        total_chunks = []

        for day in chunks_by_days:

            for chunk in chunks_by_days[day]:
                total_chunks.append(chunk)
        
        return total_chunks

    def get_closing_prices(self, chunks_by_days: dict):

        closing_prices = []
        for day in chunks_by_days:
            chunks = chunks_by_days[day]

            for chunk in chunks:
                closing_prices.append(chunk[-1])
        
        return closing_prices

    def make_stationary(self, prices: dict, get_close: bool = False):

        

        closing_prices = []
        price_df = pd.DataFrame()
        price_df['prices'] = prices
        price_df['new_prices'] = price_df['prices'].apply(np.log)

        new_prices = list(price_df['new_prices'])
        old_prices = list(price_df['prices'])

        for i in range(len(prices)):
            prices[i] = round(new_prices[i], 3)

        chunks = []
        for i in range(len(prices)):
            
            if i+10 < len(prices):
                chunks.append(prices[i:i+10])
                closing_prices.append(old_prices[i:i+10][-1])
        
        if get_close:
            return chunks, closing_prices
        else:
            return chunks

    def combine_chunks(self, price_chunks: list, vol_chunks: list, price_SPY_chunks: list, vol_SPY_chunks: list):

        total_chunks = []

        for i in range(len(price_SPY_chunks)):
            
            total_chunks.append([val for pair in zip(price_SPY_chunks[i], vol_SPY_chunks[i]) for val in pair])
            
            for k in range(len(price_chunks)):
                total_chunks[i] += [val for pair in zip(price_chunks[k][i], vol_chunks[k][i]) for val in pair]
        
        return total_chunks

    def transform_relative(self, chunks_by_days, spy: bool = False):

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

    def get_closing_daily_price(self, chunks_by_days: dict):

        closing_prices = []

        for day in chunks_by_days:
            
            closing_day = []
            chunks = chunks_by_days[day]

            for chunk in chunks:

                for i in range(len(chunk)):
                    closing_day.append(chunk[i])
            
            closing_prices.append(round((sum(closing_day)/len(closing_day)),2))
        
        return closing_prices

    def create_ema(self, day_average_list: list, num_days: int):
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

    def get_emas(self, chunks_by_days: dict, new_chunks_by_days: dict, num_of_min):

        day_average_list = []

        for day in chunks_by_days:

            chunks = chunks_by_days[day]
            day_total = 0
            day_avg = 0

            for chunk in chunks:
                for i in range(len(chunk)):
                    day_total += chunk[i]
            
            day_avg = day_total / (len(chunk) * num_of_min)
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

    def agent(self, state_shape, action_shape):
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

    def train(self, env, replay_memory, model, target_model, done):
        
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

    def save_state(self, model: object, target_model: object, it_num: int, replay_mem: deque, X: list, Y: list, max_profits: list):
        
        model.save(f"model_2_{it_num}")

        target_model.save(f"target_model_2_{it_num}")

        with open(f"replay_mem_2_{it_num}.pkl", 'wb') as f:
            pickle.dump(replay_mem, f)
        
        with open(f"X_2_{it_num}.pkl", 'wb') as f:
            pickle.dump(X, f)
        
        with open(f"Y_2_{it_num}.pkl", 'wb') as f:
            pickle.dump(Y, f)
        
        with open(f"max_profits_2_{it_num}.pkl", 'wb') as f:
            pickle.dump(max_profits, f)
        
        if it_num != 0:
            shutil.rmtree(f"model_2_{it_num-1}")
            shutil.rmtree(f"target_model_2_{it_num-1}")
            os.remove(f"replay_mem_2_{it_num-1}.pkl")
            os.remove(f"X_2_{it_num-1}.pkl")
            os.remove(f"Y_2_{it_num-1}.pkl")
            os.remove(f"max_profits_2_{it_num-1}.pkl")
        
    def simulate(self, env: TrainingEnviroment):

        epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        decay = 0.005       # rate of increasing exploitation vs exploration - Change decay rate, we have 30,000 examples but reach full optimization after 1000
        episode = 0
        total_segment_reward = 0

        X = []
        y = []
        max_profits = []

        model = self.agent((93,), 3)
        target_model = self.agent((93,), 3) # Making neural net with input layer equal to state space size, and output layer equal to action space size
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

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                    self.train(env, replay_memory, model, target_model, done)            # training the main model
                    
            
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            target_model.set_weights(model.get_weights())

            print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} -  it number: {i} - epsilon: {epsilon}")
            X.append(len(X) + 1)
            max_profits.append(env.max_profit)
            y.append(env.get_current_money())
            total_segment_reward = 0

            self.save_state(model, target_model, (episode-1), replay_memory, X, y, max_profits)
            

        X, Y = np.array(X).reshape(-1,1), np.array(y).reshape(-1,1)
        plt.plot(X, y)  # Plot the chart
        plt.xlabel("Iteration Number")
        plt.ylabel("Number of decisions made")
        plt.show()
        plt.plot( X, LinearRegression().fit(X, Y).predict(X))
        plt.show()


        with open("old_x.pkl", 'wb') as f:
            pickle.dump(X, f)

        with open("old_y.pkl", 'wb') as f:
            pickle.dump(y, f)

    def test(self, env: TrainingEnviroment):

        epsilon = 0.01 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time
        max_epsilon = 1 # You can't explore more than 100% of the time - Makes sense
        min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time - Optimize somehow?
        decay = 0.01       # rate of increasing exploitation vs exploration - Change decay rate, we have 30,000 examples but reach full optimization after 1000
        episode = 0

        total_length = int(len(env.chunks)/3)
        done = False
    
        current_money = []
        iteration = []
        action_list = []

        model = keras.models.load_model(f"cache/model_1_{850}")
        target_model = keras.models.load_model(f"cache/target_model_1_{850}") # Making neural net with input layer equal to state space size, and output layer equal to action space size
        with open(f"cache/replay_mem_1_{850}.pkl", 'rb') as f:
            replay_memory = pickle.load(f)
        
        i = 0
        env.load_year_test()

        while(not done):
            
            done = False
            steps_to_update_target_model = 0
            
            steps_to_update_target_model += 1 
            random_number = np.random.rand()
            current_state = env.get_current_state()
            

            if random_number <= epsilon:  # Explore  
                action = env.get_random_action() # Just randomly choosing an action
            
            else: #Exploitting
                current_reshaped = np.array(current_state).reshape([1, np.array(current_state).shape[0]])
                predicted = model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
                action = np.argmax(predicted) 
            
            action_list.append(action)
            reward, done = env.test_step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
            new_state = env.get_current_state()                 # is this bad? Because maybe the model is assuming its affecting the market place with actions, should proboballt just update the current money and not the actual step
            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 5 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                self.train(env, replay_memory, model, target_model, done)            # training the main model
            
            if steps_to_update_target_model % 100 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                episode += 1
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
                target_model.set_weights(model.get_weights())
            

            print(f"Current money =  ${env.get_current_money()} -  it number: {i} / {total_length} - epsilon: {epsilon}")
            current_money.append(env.get_current_money())
            iteration.append(i)
            i += 1
        
        with open(f"current_money.pkl", 'wb') as f:
            pickle.dump(current_money, f)
        with open(f"iteration.pkl", 'wb') as f:
            pickle.dump(iteration, f)
        with open(f"action_list.pkl", 'wb') as f:
            pickle.dump(action_list, f)

    def trend_analysis():

        with open("iteration.pkl", 'rb') as f:
            X = np.array(pickle.load(f))
        
        with open("SPY_prices.pkl", 'rb') as f:
            Y = np.array(pickle.load(f))


        Y = Y[int(len(Y)/3):]

        plt.plot(X, Y)
        m, b = np.polyfit(X, Y, 1)
        
        plt.plot(X, m*X+b)
        print(f"Slop: {m}")
        plt.show()
    

def main():

    trainer_model = ModelTrainer()
    chunks, closing_prices = trainer_model.create_training_chunks()
    env = TrainingEnviroment(chunks, closing_prices)
    trainer_model.simulate(env)

    #chunks, closing_prices = create_training_chunks()
    #env = Enviroment(chunks, closing_prices)
    #test(env)

   #trend_analysis()



if __name__ == '__main__':
    main()