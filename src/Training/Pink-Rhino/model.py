from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

from training_enviroment import TrainingEnviroment

class Model():

    model: object
    target_model: object
    
    def __init__(self, state_shape: int, action_shape: int) -> None:
        
        self.model = self.create_agent(state_shape, action_shape)
        self.target_model = self.create_agent(state_shape, action_shape)

    def create_agent(self, state_shape: int, action_shape: int) -> object:
        """ Takes the current state shape and the action space shape
        to create neural network paramterized for this. Yses relu activation
        and HEUniform transformation normalizer. 36 Hidden layers all together 
        
        Notes:
            The agent maps X-states to Y-actions
            e.g. The neural network output is [.1, .7, .1, .3]      # Is this the q value then?
            The highest value 0.7 is the Q-Value.
            The index of the highest action (0.7) is action #1.     # So q value for all possible actions, highest is chosen

        Arguments:
            state_shape (int): The shape of the current state space

            action_shape (int): The shape of the current action space
        
        Return:
            model (keras.neural_net): Neural network by keras
        
        Side Effects:
            None
        """

        learning_rate = 0.001       #Exploration rate
        init = tf.keras.initializers.HeUniform()        #Certain normalizer?
        model = keras.Sequential()      #Must be type of neural net?
        model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))      #Maybe this is copying over the weights
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
        return model


    def train(self, replay_memory: deque, model: object, target_model: object, done: bool) -> None:
        """ Thes the current enviroment, replay memeory, model and target model
        to test if there is enoguh memory cached. If there is, takes a random 128 
        examples from the memory and uses that to retrain the target model 
        
        Arguments:
            env (TrainingEnviroment): The current TrainingEnviroment object assoicated with the training

            replay_memory (deque): The current cached memeory associated with the training

            model (object): The given neural network

            target_model (object): The given nerual network to train

            done (bool): Whether training has finished or not
        
        Return:
            None
        
        Side Effects:
            None
        """

        learning_rate = 0.7         
        discount_factor = 0.618     

        MIN_REPLAY_SIZE = 1000      
        if len(replay_memory) < MIN_REPLAY_SIZE:        # Only do this function when we've gone through atleast 1000 steps?
            return

        batch_size = 64 * 2     # Getting random 128 batch sample from 
        mini_batch = random.sample(replay_memory, batch_size)       # Grabbing said random sample
        current_states = np.array([transition[0] for transition in mini_batch])     # Getting all the states from your sampled mini batch, because index 0 is the observation
        current_qs_list = self.model.predict(current_states)     # Predict the q values based on all the historical state
        new_current_states = np.array([transition[3] for transition in mini_batch]) # Getting all of the states after we executed our action? 
        future_qs_list = self.target_model.predict(new_current_states)       # the q values resulting in our action

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
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)      