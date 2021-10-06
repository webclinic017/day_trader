import random
from re import M
import numpy as np

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
    week: int

    def __init__(self, chunks, closing_prices, min_interval) -> None:
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
        
        self.min_interval = min_interval
        self.week = int(1655 / self.min_interval)
        self.chunks = chunks
        self.curr_chunk = random.randint(0,(len(self.chunks) - int(5000 / self.min_interval)))
        self.num_chunks = 0
        self.buy_price = -1
        self.curr_money = 1000
        self.prev_money = 1000
        self.max_profit = 1000
        self.closing_prices = closing_prices
        self.goal_profit = int(1000 * self.weekly_return())
    
    def get_current_stock_price(self):
        return self.closing_prices[self.curr_chunk]

    def weekly_return(self):

        weekly_start_prices = self.closing_prices[self.curr_chunk]
        weekly_stop_prices = self.closing_prices[self.curr_chunk+self.week]

        change = (weekly_stop_prices/weekly_start_prices)

        if change < 1:
            change = 1.005
        else:
            change += 0.005
            
        return change

    def load_year_test(self) -> None:
        """ Sets the chunks as the last third
        to test the previous year of data
        
        Arguments:
            None

        Return:
            none
        
        Side Effects:
            None
        """

        self.curr_chunk = len(self.chunks) - int(len(self.chunks)/3)        # setting initial to 2/3 the way through 3 years

    def reset(self, pos: int) -> None:
        """ Function to reset all state values of the class.
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """
        if pos > -1:
            self.curr_chunk = pos
        else:
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
        
        cash = self.curr_money
        hypothetical = 0
        
        if self.buy_price > 0:
            curr = self.closing_prices[self.curr_chunk]
            hypothetical =  1000 * curr/self.buy_price
        
        return round((cash+hypothetical), 2)
  
    def get_reward(self, current: int, decay: int) -> int:
        """ Caclualtes the reward for selling the current stock

        
        Arguments:
            current (int): The current price of the stock

            decay (int): The current decay of the training model 

        Return:
            reward (int): The profit or loss for selling the given stock
        
        Side Effects:
            None
        """
        
        reward = 0

        if self.buy_price > 0:
            
            reward = current - self.buy_price
            self.curr_money = round(1000 * (current / self.buy_price), 1)
            return round(reward, 1)

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
        curr_state = np.append(curr_state, self.buy_price)
        curr_state = np.append(curr_state, self.get_current_money())
        curr_state = np.append(curr_state, self.num_chunks)
        curr_state = np.append(curr_state, self.goal_profit)

        return curr_state

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

                if self.curr_money > 100 and len(self.buy_prices) < 10:
                    self.buy_prices.append(self.closing_prices[self.curr_chunk -1])       # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 100

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_prices:
                    past_avg = self.get_past_10_avg()
                    current_price = self.closing_prices[self.curr_chunk -1]
                    reward = current_price - past_avg  
                else:
                    reward = 0
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_prices = []
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks > (self.week - 1) and self.num_chunks % self.week == 0:        # Checking if we made 0.5 % for the week

                if self.get_current_money() < self.goal_profit:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    return -100, False              # Not ending because were testing

                else:
                    
                    self.goal_profit *= 1.005

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 100, False

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

                if self.curr_money > 100:
                    self.buy_price = self.closing_prices[self.curr_chunk -1]    # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 1000

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_price > 0:
                    past_avg = self.get_past_10_avg()
                    current_price = self.closing_prices[self.curr_chunk -1]
                    reward = current_price - past_avg  
                
                else:
                    reward = 0
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_price = 0
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks > (self.week - 1) and self.num_chunks % self.week == 0:        # Checking if we made 1 % for the week

                if self.get_current_money() < self.goal_profit:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    return -100, False, self.curr_chunk - self.week

                else:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 100,  True, -1

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward,  False, -1
        
       
        else:
            self.curr_chunk = random.randint(0,(len(self.chunks) - int(5000 / self.min_interval)))          # Wrapping around to new random location
            return 0, True, -1
            
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




