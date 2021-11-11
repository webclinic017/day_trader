from textwrap import fill
import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
import alpaca_trade_api as tradeapi
from datetime import date, datetime, timedelta
import pickle
import os.path
import os
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
import shutil
import matplotlib.pyplot as plt
from dateutil import parser
from training_enviroment import TrainingEnviroment
import sys
from model import Model
import time
import threading


class ModelTrainer():

    models: Model

    def __init__(self) -> None:

        self.models = Model()
   
    def save_state(self, model: object, target_model: object, it_num: int, replay_mem: deque, X: list, goal_diff: list, Y: list, max_profits: list, ver: int) -> None:
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
        
        with open(f"models/{ver}/goal_diff_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(goal_diff, f)
        
        with open(f"models/{ver}/Y_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(Y, f)
        
        with open(f"models/{ver}/max_profits_{ver}_{it_num}.pkl", 'wb') as f:
            pickle.dump(max_profits, f)
        
        # Train on same month over and over? 
        if it_num != 0:
            shutil.rmtree(f"models/{ver}/model_{ver}_{it_num-1}")
            shutil.rmtree(f"models/{ver}/target_model_{ver}_{it_num-1}")
            os.remove(f"models/{ver}/replay_mem_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/X_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/goal_diff_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/Y_{ver}_{it_num-1}.pkl")
            os.remove(f"models/{ver}/max_profits_{ver}_{it_num-1}.pkl")
        
    def simulate(self, env: TrainingEnviroment, decay: float, ver: int, training_iterations: int) -> None:
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
        goal_diff = []

        self.models = Model((1, 134), 3)
        self.models.target_model.set_weights(self.models.model.get_weights())
        
        replay_memory = deque(maxlen=100_000)



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
                    predicted = list(self.models.model.predict(current_reshaped).flatten())          # Predicting best action, not sure why flatten (pushing 2d into 1d)
                    action = np.argmax(predicted) 
                    action += 1
                
                reward, done = env.step(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state
                
                new_state = current_state               
                new_state[-2] = env.num_chunks
                new_state[-3] = env.get_current_money()
                new_state[-4] = env.buy_price
                
                replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory

                # 3. Update the Main Network using the Bellman Equation, can maybe do this for every cpu we have and paralize the training process
                if steps_to_update_target_model % 50 == 0 or done:                   # If we've done 4 steps or have lost/won, updat the main neural net, not target
                    self.models.train(replay_memory, done)            # training the main model
                        
                
            episode += 1
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
            self.models.target_model.set_weights(self.models.model.get_weights())

            print(f"Made it to ${env.get_current_money()} - Max Money: {env.max_profit} - Goal Profit: {env.goal_profit} -  it number: {i} - epsilon: {epsilon}")
            X.append(len(X) + 1)
            goal_diff.append(env.get_current_money() - env.goal_profit)
            max_profits.append(env.max_profit)
            y.append(env.get_current_money())
            total_segment_reward = 0

            self.save_state(self.models.model, self.models.target_model, (episode-1), replay_memory, X, goal_diff, y, max_profits, ver)
        
    def train_from_save(self, env: TrainingEnviroment, iteration: int, model_name: str, target_model_name: str, replay_mem_name: str, epsilon: float, decay: int, ver: int, max_it: int):
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


        for i in tqdm(range(max_it - iteration)):

            done = False
            steps_to_update_target_model = 0
            env.reset()
            monies = []

            while(not done):

                total_segment_reward += 1
                steps_to_update_target_model += 1 
                random_number = np.random.rand()
                current_state = env.get_current_state()

                if random_number <= epsilon:  # Explore  
                    action = env.get_random_action() # Just randomly choosing an action
                
                else: #Exploitting
                    
                    current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
                    predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
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

                monies.append(env.get_current_money())
                with open(f"models/{ver}/monies.pkl", 'wb') as f:
                    pickle.dump(monies, f)

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

    def test_v2(self, env: TrainingEnviroment, model_name: str, ver: int, p_num: int, shared_list: list = []) -> None:
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
        print("In here")
        # Can compute this in parallel because everything is reset 
        epsilon = 0.01 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start - This decreases over time

        total_length = int(len(env.chunks))
        overal_done = False
    
        action_list = []

        self.models.model = keras.models.load_model(f'models/{model_name}/model_{model_name}_{ver}')
        self.models.target_model = keras.models.load_model(f'models/{model_name}/target_model_{model_name}_{ver}') # Making neural net with input layer equal to state space size, and output layer equal to action space size
        with open(f'models/{model_name}/replay_mem_{model_name}_{ver}.pkl', 'rb') as f:
            replay_memory = pickle.load(f)
        
        i = 0
        money_made = 0
        monthly_info = {
            "buy_pos": {},
            "sell_pos": {},
            "model": [],
            "coin": []
        }
        months = []

        base_price = env.get_current_stock_price()
        
        while(not overal_done):
            
            done = False
            
            current_state = env.get_current_state()
            monthly_info["coin"].append(1000 * (env.get_current_stock_price() / base_price))

            current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
            predicted = self.models.model.predict(current_reshaped).flatten()           # Predicting best action, not sure why flatten (pushing 2d into 1d)
            action = np.argmax(predicted) 
            action += 1
            
            action_list.append(action)

            if action == 1 and env.buy_price  == -1:
                monthly_info["buy_pos"][i] = list(predicted)
            
            elif action == 3 and env.buy_price > 0:
                monthly_info["sell_pos"][i] = list(predicted)
                
            reward, weekly_done, overal_done = env.test_step_v2(action, current_state, epsilon)      # Executing action on current state and getting reward, this also increments out current state

            new_state = current_state               
            new_state[-2] = env.num_chunks
            new_state[-3] = env.get_current_money()
            new_state[-4] = env.buy_price

            replay_memory.append([current_state, action, reward, new_state, done])      # Adding everything to the replay memory


            print(f"Current money =  ${env.get_current_money()*1} -  it number: {i} / {total_length} - epsilon: {epsilon}")
            monthly_info["model"].append(env.get_current_money())

            i += 1

            if weekly_done:
                print(f"We didn't meet goal profit, made it to: {env.get_current_money()*1} , should have made it to: {env.goal_profit*1}")
                money_made += (env.get_current_money()*1) - 1000
                print(f"Current made money: {money_made}")

                i = 0

                months.append(monthly_info)
                monthly_info = {
                    "buy_pos": {},
                    "sell_pos": {},
                    "model": [],
                    "coin": []
                }

                env.goal_profit = 1000 * env.weekly_return()
                env.curr_money = 1000
                env.buy_price = -1
                env.prev_money = 1000
                env.max_profit = 1000
                env.num_chunks = 0
        
        shared_list.append(money_made)

        with open(f"models/{model_name}/months_{ver}_{p_num}.pkl", 'wb') as f:
            pickle.dump(months, f)
        with open(f"models/{model_name}/v2_action_list_{ver}_{p_num}.pkl", 'wb') as f:
            pickle.dump(action_list, f)
        
    def test(self, env: TrainingEnviroment, model_name: str, target_model_name: str, replay_mem_name: str, ver: int, shared_list: list) -> None:
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

        total_length = len(env.chunks)
        done = False
    
        current_money = []
        iteration = []
        action_list = []

        self.models.model = keras.models.load_model(model_name)
        self.models.target_model = keras.models.load_model(target_model_name) # Making neural net with input layer equal to state space size, and output layer equal to action space size
        with open(replay_mem_name, 'rb') as f:
            replay_memory = pickle.load(f)
        
        i = 0
        base_money = []
        buy_positions = []
        sell_positions = []
        base_price = env.get_current_stock_price()


        while(not done):
            
            done = False
            steps_to_update_target_model = 0
            
            steps_to_update_target_model += 1 
            random_number = np.random.rand()
            current_state = env.get_current_state()
            base_money.append(1000 * (env.get_current_stock_price() / base_price))

            if random_number <= epsilon:  # Explore  
                action = env.get_random_action() # Just randomly choosing an action
            
            else: #Exploitting
                current_reshaped = np.array(current_state).reshape([1, 1, len(current_state)])
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
        
        with open(f"models/1/current_money_{ver}.pkl", 'wb') as f:
            pickle.dump(current_money, f)
        with open(f"models/1/buy_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(buy_positions, f)
        with open(f"models/1/sell_pos_{ver}.pkl", 'wb') as f:
            pickle.dump(sell_positions, f)
        with open(f"models/1/base_money_{ver}.pkl", 'wb') as f:
            pickle.dump(base_money, f)
        with open(f"models/1/iteration_{ver}.pkl", 'wb') as f:
            pickle.dump(iteration, f)
        with open(f"models/1/action_list_{ver}.pkl", 'wb') as f:
            pickle.dump(action_list, f)


        shared_list.append(env.get_current_money())

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
        
    def test_trend_analysis(self, model: str, ver: str) -> None:
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

        
        with open(f"models/{model}/iteration_{ver}.pkl", 'rb') as f:
            X = list(pickle.load(f))
        with open(f"models/{model}/base_money_{ver}.pkl", 'rb') as f:
            Y_base = list(pickle.load(f))
        with open(f"models/{model}/current_money_{ver}.pkl", 'rb') as f:
            Y = list(pickle.load(f))
        with open(f"models/{model}/buy_pos_{ver}.pkl", 'rb') as f:
            buy_pos = list(pickle.load(f))
        with open(f"models/{model}/sell_pos_{ver}.pkl", 'rb') as f:
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

    def buy_confidence_analysis(self, model: str, it: str):
        
        
        with open(f"models/{model}/v2_sell_pos_{it}.pkl", 'rb') as f:
            total_sell = list(pickle.load(f))
        with open(f"models/{model}/v2_buy_pos_{it}.pkl", 'rb') as f:
            total_buy = list(pickle.load(f))
        with open(f"models/{model}/v2_total_profit_{it}.pkl", 'rb') as f:
            total_profit = list(pickle.load(f))

        buy_profit = []
        buy_loss = []

        print(total_profit)

        for i in range(len(total_sell)):
            
            if total_profit[i] > 0:

                for pos in total_buy[i]:
                    buy_profit.append(total_buy[i][pos][0])

            else:

                for pos in total_buy[i]:
                    buy_loss.append(total_buy[i][pos][0])

        
        print(f"Average buy confidence for profitable: {sum(buy_profit)/len(buy_profit)}")
        print(f"Average buy confidence for loss: {sum(buy_loss)/len(buy_loss)}")

def test_parallel(ver: str, model: str, it: str):

    pool = mp.Pool()
    shared_list = mp.Manager().list()
    env = TrainingEnviroment(6, {}, False, 60)

    mt = ModelTrainer()

    for i in range(os.cpu_count()):

        mt.test_v2(env, "1.1", 99, i, shared_list)
        if ver == "v2":
            pool.apply_async(func=mt.test_v2, args=(env, "1.1", 99, i, shared_list,))
        
        elif ver == "v1":
            pool.apply_async(func=mt.test, args=(env, f'models/{model}/model_{model}_{it}', f'models/{model}/target_model_{model}_{it}', f'models/{model}/replay_mem_{model}_{it}.pkl', 99, i, ))


    pool.close()
    pool.join()

    print(f"Average yearly return: {int(sum(shared_list)/len(shared_list))}")

def create_env(coin, training_bool):

    env = TrainingEnviroment(coin, {}, training_bool, 60)

def main():

    env = TrainingEnviroment(6, {}, False, 60)
    mt = ModelTrainer()
    #mt.simulate(env, 0.01, "1.1", 100)

    #env = TrainingEnviroment(6, {}, False, 60)
    mt.test_v2(env, '1.1', 99)

   # mt.test_trend_analysis("1", "99")

    #test_parallel("v2", "1.1", "99")
    #mt.buy_confidence_analysis("1", "99")
    
   

    
# Figure out way to reinvest the money, should do this in training too
# May need to run over a month now due to the increase or whatever
    

if __name__ == '__main__':
    main()