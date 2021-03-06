import random
import multiprocessing as mp
from multiprocessing import Process
import os
import operator
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

from model import Model
import model_trainer
from training_enviroment import TrainingEnviroment

# Replace with negative one in shared dict on the training side, so the new process knows to index into there

def find_new_folder(completed: list):

    all_folders = list(os.listdir("models"))
    all_folders = [x for x in all_folders if "test" in str(x)]

    for folder in all_folders:
        
        if folder not in completed:
            completed.append(folder)
            return folder
    
    return ""


def live_graph():
    

    print("In here")
    figure, axis = plt.subplots(2, 4)
    completed = []

    folders = {
        0: find_new_folder(completed),
        1: find_new_folder(completed),
        2: find_new_folder(completed),
        3: find_new_folder(completed),
        4: find_new_folder(completed),
        5: find_new_folder(completed),
        6: find_new_folder(completed),
        7: find_new_folder(completed)
    }

    f_axis = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 2),
        3: (0, 3),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (1, 3)
    }

    while(True):

        for i in range(8):
            
            if not folders[i]:
                folders[i] = find_new_folder(completed)
                continue

            with open(f"models/{folders[i]}/Y.pkl", 'rb') as f:
                y = list(pickle.load(f))
            with open(f"models/{folders[i]}/max_profits.pkl", 'rb') as f:
                y_max = list(pickle.load(f))
            x = list(range(len(y)))

            if -1 in y:
                folders[i] = find_new_folder(completed)
                axis[f_axis[i][0], f_axis[i][1]].clear()
                continue

            axis[f_axis[i][0], f_axis[i][1]].plot(x, y, color='blue')
            axis[f_axis[i][0], f_axis[i][1]].plot(x, y_max, color='red')
            axis[f_axis[i][0], f_axis[i][1]].set_title(folders[i])

        plt.pause(1)


class Evolution:
    
    population: list
    pop_size: int
    cache: list

    model_performance: dict
    average_performance: list

    current_gen: int

    max_first: int
    max_second: int


    def __init__(self, max_first, max_second) -> None:
        
        self.max_first = max_first
        self.max_second = max_second

        self.current_gen = 0
        self.average_performance = []

        self.pop_size = 90
        self.population = []
        self.model_performance = mp.Manager().dict()

        for i in range(90):
            first = random.randint(int(max_first*0.1), max_first)
            second = random.randint(int(max_second*0.1), max_second)
            self.population.append([first, second])

    def save_test(self):

        with open(f"models/test_{self.current_gen}.pkl", "wb") as f:
            pickle.dump(self.model_performance, f) 

    def test_yearly(self):

        pool = mp.Pool(processes=os.cpu_count())

        for model_name in self.model_performance:
            
            pool.apply_async(
                    model_trainer.test, 
                    args=(
                            model_name,
                            self.model_performance,
                        )
                    )
            
        pool.close()
        pool.join()

        self.save_test()
    
    def test_weekly(self):

        pool = mp.Pool(processes=os.cpu_count())

        for model_name in self.model_performance:
            
            pool.apply_async(
                    model_trainer.test_v2, 
                    args=(
                            model_name,
                            self.model_performance,
                        )
                    )
            
        pool.close()
        pool.join()

        self.save_test()

    def train(self):

        pool = mp.Pool(processes=os.cpu_count())

        for i in range(len(self.population)):
            
            self.model_performance[f"test_{self.current_gen}_{i}"] = []
            pool.apply_async(
                    model_trainer.simulate, 
                    args=(
                            0.01, 
                            f"test_{self.current_gen}_{i}", 
                            100, 
                            self.population[i]
                        )
                    )
            
            
        p = mp.Process(target=live_graph, args=())
        p.start()

        pool.close()
        pool.join()

        p.terminate

    def pop_random(self, lst):
        
        index = random.randrange(0, len(lst))
        return lst.pop(index)

    def get_top_30(self):

        for model in self.model_performance:
            self.model_performance[model] = max(self.model_performance[model])
        
        ranked = dict(sorted(self.model_performance.items(), key=operator.itemgetter(1), reverse=True))
       
        self.average_performance.append(sum(list(ranked.values())[0:30])/30)
        ranked = list(ranked.keys())[0:30]

        architectures = []

        for model in ranked:
            index = int(str(model).split("_")[-1])
            architectures.append(self.population[index])
        
        return architectures

    def create_pairs(self, top_30: list):

        pairs = []
        copy_pop = top_30

        for i in range(15):
           
            p1 =  self.pop_random(copy_pop)
            p2 =  self.pop_random(copy_pop)

            pairs.append([p1,p2])
        
        return pairs

    def crossover(self, top_30: list):
        
        first = 0
        second = 1
        pairs = self.create_pairs(top_30)

        children = []

        for pair in pairs:

            p1 = pair[first]
            p2 = pair[second]

            c_first = 0
            c_second = 0

            q = random.randint(0,100)

            if p1[0] * (q/100) >= p2[0]:
                c_first = (p1[0] * (q/100)) - ((1 - (q/100)) * p2[0])
            
            else:
                c_first = (p1[0] * (q/100)) + ((1 - (q/100)) * p2[0])
            
            if p1[1] * (q/100) >= p2[1]:
                c_second = (p1[1] * (q/100)) - ((1 - (q/100)) * p2[1])
            
            else:
                c_second = (p1[1] * (q/100)) + ((1 - (q/100)) * p2[1])
            
            children.append([c_first, c_second])
        
        return children

    def mutate(self, combined: list):

        mutated = []

        for model_arch in combined:

            first = model_arch[0]
            second = model_arch[1]

            mutated_first = random.randint(int(first - (0.2*first)), int(first + (0.2*first)))
            mutated_second = random.randint(int(second - (0.2*second)), int(second + (0.2*second)))

            mutated.append([mutated_first, mutated_second])

        return mutated
    
    def cont(self):

        x = []

        for i in range(len(self.average_performance)):
            x.append(i)

        m, b = np.polyfit(x, self.average_performance, 1)

        return m > 0

    def evolve(self):
        
        finished = False

        while(not finished):

            self.train()
            self.test_yearly()
            self.test_weekly()

            top_30 = self.get_top_30()

            children = self.crossover(top_30)
            combined = top_30 + children
            mutated = self.mutate(combined)

            new_population = combined + mutated
            self.population = new_population
            
            print(f"Average Top 30 Pop Return: {self.average_performance[self.current_gen]}")
            self.current_gen += 1

            finished = self.cont()

       


def main():
    
    evolver = Evolution(515, 370)
    evolver.evolve()


if __name__ == '__main__':
    main()