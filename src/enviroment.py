import random
import numpy as np

class Enviroment:
    
    chunks: list
    curr_chunk: int
    num_chunks: int
    buy_prices: list
    curr_money: int
    closing_prices: list
    max_profit: int
    goal_profit: int

    def __init__(self, chunks, closing_prices):
        
        self.chunks = chunks
        self.curr_chunk = random.randint(0,(len(self.chunks) - 5000))
        self.num_chunks = 0
        self.buy_prices = []
        self.curr_money = 1000
        self.prev_money = 1000
        self.max_profit = 1000
        self.goal_profit = 1005
        self.closing_prices = closing_prices
    
    def load_year_test(self):
        self.curr_chunk = len(self.chunks) - int(len(self.chunks)/3)        # setting initial to 2/3 the way through 3 years

    def reset(self):
        self.curr_chunk = random.randint(0,(len(self.chunks) - 5000))
        self.buy_prices = []
        self.curr_money = 1000
        self.prev_money = 1000
        self.max_profit = 1000
        self.goal_profit = 1005
        self.num_chunks = 0

    def get_past_10min_avg(self):

        past_10min = self.closing_prices[self.curr_chunk-10:self.curr_chunk]
        total = sum(past_10min)
        avg = total/len(past_10min)

        return avg

    def get_current_money(self):
        
        cash = self.curr_money
        hypothetical = 0

        for buy_price in self.buy_prices:
            hypothetical += (100 * (self.closing_prices[self.curr_chunk] / buy_price))
        
        return round((cash+hypothetical), 2)
  
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
        curr_state = np.append(curr_state, len(self.buy_prices))
        curr_state = np.append(curr_state, self.curr_money)
        curr_state = np.append(curr_state, self.num_chunks)
        curr_state = np.append(curr_state, self.goal_profit)

        return curr_state

    def test_step(self, action, prev_chunk, decay):
       
        self.num_chunks += 1

        if self.curr_chunk + 1 < int(len(self.chunks)):
 
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.curr_money > 100:
                    self.buy_prices.append(self.closing_prices[self.curr_chunk -1])       # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 100

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_prices:
                    past_avg = self.get_past_10min_avg()
                    current_price = self.closing_prices[self.curr_chunk -1]
                    reward = current_price - past_avg  
                else:
                    reward = 0
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_prices = []
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks > 1654 and self.num_chunks % 1655 == 0:        # Checking if we made 1 % for the week

                if self.get_current_money() < self.goal_profit:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    return -100, False  # Not ending it if we don't make quota, just penalizing

                else:
                    
                    self.goal_profit *= 1.005

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 100,  False

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward,  False
        
       
        else:
            return 0, True # hit the end, ending the simulation

    def step(self, action, prev_chunk, decay):
        
        self.num_chunks += 1

        if self.curr_chunk + 1 < int(len(self.chunks)):
 
            self.curr_chunk += 1
            reward = 0

            if action == 1:     # Buying a share of the stock

                if self.curr_money > 100:
                    self.buy_prices.append(self.closing_prices[self.curr_chunk -1])       # Appending current price we bought the stock at for previous chunk 
                    self.curr_money -= 100

                reward = 0      # maybe adjust to encourage model to buy stocks if it's a problem buying stocks 
                #print(f"     Decided to buy stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")
                

            elif action == 2:   # Holding the stock - CAN ADD REWARD FOR HOLDING WHEN GOING UP AND HOLDING WHEN GOING DOWN

                if self.buy_prices:
                    past_avg = self.get_past_10min_avg()
                    current_price = self.closing_prices[self.curr_chunk -1]
                    reward = current_price - past_avg  
                else:
                    reward = 0
                #print(f"     Decided to hold stock || {self.get_current_money()} || {self.curr_chunk} \n")

            elif action == 3:   # Selling the stock
                reward = self.get_reward(self.closing_prices[self.curr_chunk -1], decay)        # Selling based on price that the model has seen and is acting on
                self.buy_prices = []
                #print(f"     Decided to sell stock at price: {self.closing_prices[self.curr_chunk -1]} || {self.get_current_money()} || {self.num_chunks} \n")

            
            if self.num_chunks > 1654 and self.num_chunks % 1655 == 0:        # Checking if we made 1 % for the week

                if self.get_current_money() < self.goal_profit:

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()
                    
                    return -100, True

                else:
                    
                    self.goal_profit *= 1.005

                    if self.get_current_money() > self.max_profit:
                        self.max_profit = self.get_current_money()

                    return 100,  False

            if self.get_current_money() > self.max_profit:
                self.max_profit = self.get_current_money()

            return reward,  False
        
       
        else:
            self.curr_chunk = random.randint(0,(len(self.chunks) - 5000))          # Wrapping around to new random location
            return self.step(action, prev_chunk, decay)
            
    def get_random_action(self):
        return random.randint(1,3)

    def begin(self):
        self.curr_chunk += 1
        return self.chunks[0]
