import numpy as np
from tensorflow import keras
from collections import deque
import random
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
import pickle
import os.path
import os
import matplotlib.pyplot as plt
import websocket, json
import time
from threading import Thread
import logging
import ecs_logging


class LiveModel():

    api: tradeapi.REST
    logger: logging.Logger


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

    def __init__(self, it: int) -> None:
        """ Base initializer function. Creates new ALPACA trading API,
        initializes the prev_10 containers as deque's as length of 10, 
        loads in previous models and sets montetary trackers and goals.

        Arguments:
            it (int): The iteration number of the model to load

        Return:
            None
        
        Side Effects:
            None
        """
        
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('day_trader_logging.json')
        handler.setFormatter(ecs_logging.StdlibFormatter())
        self.logger.addHandler(handler)

        self.logger.info("Init", extra={"http.request.body.content": "Starting Live Model"})

        self.api = tradeapi.REST()

        self.prev_10_BND = deque(maxlen=10)
        self.prev_10_SPY = deque(maxlen=10)
        self.prev_10_VTI = deque(maxlen=10)
        self.prev_10_VXUS = deque(maxlen=10)

        try:
            self.model = keras.models.load_model(f"cache/(LIVE)model_1_{it}")
            self.target_model = keras.models.load_model(f"cache/(LIVE)target_model_1_{it}")
            with open(f"cache/(LIVE)replay_mem_1_{it}.pkl", 'rb') as f:
                self.replay_memory = pickle.load(f)
        
        except:
            self.logger.critical("Init", extra={"http.request.body.content": "Mddles were not loaded"})
            exit()
        
        
        self.buy_prices = []
        self.curr_money = float(self.api.get_account().equity)
        self.goal_money = self.curr_money * 1.05
        self.mins_into_week = 0

    def  start_action_loop(self) -> None:
        """ Creates continuous loop to firstly check if 
        we have 10min of history yet, if we do, calls
        our model on our current input space. Does this once a min.

        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """

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
  
    def fill_remaining(self) -> None:
        """ Takes most recent value and repeats it in dequeue
        until it is filled. This is under the assumption the 
        lack of incoming data for the stock is due to the price
        not changing.
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """
        
        self.logger.info("fill_remaining", extra={"http.request.body.content": "Filling the deques to 10"})

        for i in range(10 - len(self.prev_10_VXUS)):
            self.prev_10_VXUS.append(self.prev_10_VXUS[len(self.prev_10_VXUS) - 1])
        
        for i in range(10 - len(self.prev_10_BND)):
            self.prev_10_BND.append(self.prev_10_BND[len(self.prev_10_BND) - 1])
        
        for i in range(10 - len(self.prev_10_VTI)):
            self.prev_10_VTI.append(self.prev_10_VTI[len(self.prev_10_VTI) - 1])

    def get_random_action(self) -> int:
        """ Gets a random numbber between [1,3], inclusive
        
        Arguments:
            None

        Return:
            random_action (int): Random number between 1 and 3
        
        Side Effects:
            None
        """

        return random.randint(1,3)
    
    def get_10min_avg(self) -> int:
        """ Gets the 10min average of the SPY stock
        
        Arguments:
            None

        Return:
            average (int): Average price of SPY stock over past 10min
        
        Side Effects:
            None
        """

        total = sum(self.prev_10_SPY[0:9])
        return total/9

    def execute_buy(self) -> None:
        """ Buys 100$ worth of SPY stock, executed using 
        API
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """

        try:
            self.api.submit_order(symbol="SPY", notional=100.00)
            self.logger.info("execute_buy", extra={"http.request.body.content": "Executed buy"})

        except:
            self.logger.critical("execute_buy", extra={"http.request.body.content": "Unable to execute buy"})
        
    def train(self, replay_memory, model, target_model, done) -> None:
        """ If the replay memeory is over 1000 entries, the model 
        retrains itself on a random subset of 128 and transfer the
        model weihts over to the target model.
        
        Arguments:
            None

        Return:
           None
        
        Side Effects:
            None
        """

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

    def execute_sell(self) -> float:
        """ Executes selling all current holdings of SPY
        stock using ALPACA API
        
        Arguments:
            None

        Return:
            equity (float): The updated equity of the acount
        
        Side Effects:
            None
        """
        try:
            self.api.close_position("SPY")
            equity = float(self.api.get_account().equity)
            self.curr_money = equity

            self.logger.info("execute_sell", extra={"http.request.body.content": "Executed sell"})

            return equity
        
        except:
            self.logger.critical("execute_buy", extra={"http.request.body.content": "Unable to execut sell"})
            return -1

    def execute_action(self, action: int) -> None:
        """ Executes the decided action from the model. If 1,
        the model buys $100 of SPY stock. If 2, the model holds and 
        does nothing. If 3, the model sells all current holdings of the
        SPY stock.
        
        Arguments:
            action (int): The given action from the model

        Return:
            None
        
        Side Effects:
            None
        """
  
        reward = 0

        if action == 1:

            if self.curr_money > 100:
                self.buy_prices.append(self.prev_10_SPY[9])
                self.execute_buy()
                self.curr_money -= 100
                print("Decided to buy")
                self.logger.info("execute_action", extra={"http.request.body.content": "Buying"})


        elif action == 2:
            
            if self.buy_prices:
                past_avg = self.get_10min_avg()
                reward = self.prev_10_SPY[9] - past_avg
                print("Decided to hold")
                self.logger.info("execute_action", extra={"http.request.body.content": "Holding"})

        elif action == 3:
            old_money = self.curr_money
            new_money = self.execute_sell()
            reward = new_money - old_money
            print("Decided to sell")
            self.logger.info("execute_action", extra={"http.request.body.content": "Selling"})


        return reward

    def make_decision(self) -> None:
        """ Controls the enviroment given to the model for a 
        decision. It's called once a minute and gathers the
        past 10 min history from the SPY, VXUS, VTI and BND stocks. 
        Then also gathers the current stocks being held, the equity,
        how far into the week the model is and the goal profit.
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """

        curr_state = np.array(self.prev_10_SPY)
        curr_state = np.append(curr_state, self.prev_10_VTI)
        curr_state = np.append(curr_state, self.prev_10_VXUS)
        curr_state = np.append(curr_state, self.prev_10_BND)

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
        print("ACTION: " + str(action))
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

def main():
    model_live = LiveModel()
    return

if __name__ == '__main__':
    main()