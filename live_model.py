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
import websocket, json
import time
from threading import Thread


class LiveModel():

    api: tradeapi.REST
    auth_data: dict
    socket_url: str
    ws: websocket
    
    prev_10_SPY: deque
    prev_10_VTI: deque
    prev_10_BND: deque
    prev_10_VXUS: deque
    
    buy_prices: list
    curr_money: int
    goal_money: int
    mins_into_week: int

    epsilon: float = 0.01 
    max_epsilon: int = 1 
    min_epsilon : float= 0.01 
    decay: float = 0.01 

    model: object
    target_model: object   
    replay_memory: deque

    current_money: list = []
    iteration: list = []
    action_list: list = [] 

    STOP_SIG: bool = False


    # Model should only care about one trade at a time
    # We can just have 10 separate models going to help negate risk
    # If we 1000, each model is in charge of 100
    # can maybe make an ensable model and use a neural net to choose the best trade
    # Can easily train this 

    # Or maybe just do multiple models and choose the one that resulted in the most profit, kind of like evolution

    # DONT FORGET TO MAKE EVERYTHING RELATIVE
    # Fuck all that, give the input the array of buy prices, 0's if none exist, then the  model can output an array of 1's and 0's at each index position to dictate whether to sell or not
    # Fixing the state change in training, just updating prices and shit, the price wouldn't move, should allow the model to fiure out that it can't sell something that doesnt exist
    def __init__(self):
        
        self.api = tradeapi.REST()
        self.auth_data = {
            "action": "authenticate",
            "data": {"key_id": os.environ.get("APCA_API_KEY_ID", None), "secret_key": os.environ.get("APCA_API_SECRET_KEY", None)}
        }
        self.socket_url = "wss://data.alpaca.markets/stream"
        self.ws = websocket.WebSocketApp(self.socket_url , on_open=self.run, on_message=self.update_prices, on_close=self.close)
        
        self.prev_10_BND = deque(maxlen=10)
        self.prev_10_SPY = deque(maxlen=10)
        self.prev_10_VTI = deque(maxlen=10)
        self.prev_10_VXUS = deque(maxlen=10)

        self.buy_prices = []
        self.curr_money = 0
        self.goal_money = 0
        self.mins_into_week = 0

        self.model = keras.models.load_model(f"cache/model_1_{850}")
        self.target_model = keras.models.load_model(f"cache/target_model_1_{850}")
        with open(f"cache/replay_mem_1_{850}.pkl", 'rb') as f:
            self.replay_memory = pickle.load(f)

        self.ws.run_forever()

    def close(self):

        print(f"Closed connection")

    def  start_action_loop(self):

        while(not self.STOP_SIG):

            if (len(self.prev_10_VTI) == 10 and len(self.prev_10_BND) == 10
                and len(self.prev_10_SPY) == 10 and len(self.prev_10_VXUS) == 10):
                print(" all are full, executing decision")
                self.make_decision()
            
            else:
                print("     Not full yet")
                print(f"    SPY: {len(self.prev_10_SPY)}")
                print(f"    VTI: {len(self.prev_10_VTI)}")
                print(f"    VXUS: {len(self.prev_10_VXUS)}")
                print(f"    BND: {len(self.prev_10_BND)}")
            
            time.sleep(60)

    def update_prices(self, ws: websocket, message: dict):
        
        message = eval(message)             # Security risk - Fix later
        print(message)

        if message["stream"] == "AM.VTI":
            curr_VTI_price = message["data"]["c"]
            self.prev_10_VTI.append(curr_VTI_price)
            
        
        elif message["stream"] == "AM.VXUS":
            curr_VXUS_price = message["data"]["c"]
            self.prev_10_VXUS.append(curr_VXUS_price)
            
        
        elif message["stream"] == "AM.BND":
            curr_BND_price = message["data"]["c"]
            self.prev_10_BND.append(curr_BND_price)
            
        
        elif message["stream"] == "AM.SPY":
            curr_SPY_price = message["data"]["c"]
            self.prev_10_SPY.append(curr_SPY_price)

            if len(self.prev_10_SPY) == 10:
                self.fill_remaining()
    
    def fill_remaining(self):

        if len(self.prev_10_VXUS) > 0:
            for i in range(10 - len(self.prev_10_VXUS)):
                self.prev_10_VXUS.append(self.prev_10_VXUS[len(self.prev_10_VXUS) - 1])
        
        if len(self.prev_10_BND) > 0:
            for i in range(10 - len(self.prev_10_BND)):
                self.prev_10_BND.append(self.prev_10_BND[len(self.prev_10_BND) - 1])
        
        if len(self.prev_10_VTI) > 0:
            for i in range(10 - len(self.prev_10_VTI)):
                self.prev_10_VTI.append(self.prev_10_VTI[len(self.prev_10_VTI) - 1])

    def get_random_action(self):
        return random.randint(1,3)
    
    def get_10min_avg(self):
        
        total = sum(self.prev_10_SPY[0:9])
        return total/9

    def execute_buy(self):
        #Implement
        self.api.submit_order(symbol="SPY", notional=100.00)
        self.buy_prices.append(self.prev_10_SPY[9])     # Need to make dict and add in id?
        return

    def train(self, replay_memory, model, target_model, done):
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

    def execute_sell(self):

        self.api.close_position("SPY")
        equity = self.api.get_account().equity
        self.curr_money = equity

        return equity


    # WE SHOULD INCLUDE EQUITY AS MODEL INPUT
    def execute_action(self, action: int):
  
        reward = 0

        if action == 1:

            if self.curr_money > 100:
                self.buy_prices.append(self.prev_10_SPY[9])
                self.execute_buy()
                self.curr_money -= 100


        elif action == 2:
            
            if self.buy_prices:
                past_avg = self.get_10min_avg()
                reward = self.prev_10_SPY[9] - past_avg

        elif action == 3:
            old_money = self.curr_money
            new_money = self.execute_sell()
            reward = new_money - old_money

        return reward

    def make_decision(self):

        curr_state = self.prev_10_SPY + self.prev_10_VTI + self.prev_10_VXUS + self.prev_10_BND
        curr_state = np.copy(curr_state)
        curr_state = np.append(curr_state, len(self.buy_prices))
        curr_state = np.append(curr_state, self.curr_money)
        curr_state = np.append(curr_state, self.mins_into_week)
        curr_state = np.append(curr_state, self.goal_money)

        self.mins_into_week += 1

        if self.mins_into_week > 1499 and self.mins_into_week % 1500:       # Make this more sophisticated by checking actual dates
            self.mins_into_week = 0
        
        random_number = np.random.rand()

        if random_number <= self.epsilon:  # Explore  
            action = self.get_random_action() # Just randomly choosing an action
        else:
            current_reshaped = np.array(curr_state).reshape([1, np.array(curr_state).shape[0]])
            predicted = self.model.predict(current_reshaped).flatten()
            action = np.argmax(predicted) 

        self.action_list.append(action)
        reward = self.execute_action(action)
        
        new_state = curr_state
        new_state[-3] = self.curr_money
        new_state[-4] = len(self.buy_prices)

        self.replay_memory.append([curr_state, action, reward, new_state, False])

        if self.mins_into_week % 5 == 0:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
            self.train(self.replay_memory, self.model, self.target_model, False)    

        if self.mins_into_week % 100 == 0:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
            self.mins_into_week += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.mins_into_week)
            self.target_model.set_weights(self.model.get_weights())
        
        self.current_money.append(self.curr_money)
        self.iteration.append(self.mins_into_week)


    def run(self, ws: websocket):

        ws.send(json.dumps(self.auth_data))
        listen_message = {"action": "listen", "data": {"streams": ["AM.SPY", "AM.VXUS", "AM.VTI", "AM.BND"]}}
        ws.send(json.dumps(listen_message))

        thread = Thread(target = self.start_action_loop, args = ())
        thread.start()

        




def main():
    model_live = LiveModel()
    return


if __name__ == '__main__':
    main()