import random
import multiprocessing as mp
import os

from model import Model
import model_trainer
from training_enviroment import TrainingEnviroment



def hello():
    print("hello")

class Evolution:
    
    population: list
    pop_size: int
    cache: list

    pop_stats: dict

    max_first: int
    max_second: int


    def __init__(self, max_first, max_second) -> None:
        
        self.max_first = max_first
        self.max_second = max_second

        self.pop_size = 90
        self.population = []
        self.pop_stats = mp.Manager().dict()

        for i in range(90):
            first = random.randint(int(max_first*0.1), max_first)
            second = random.randint(int(max_second*0.1), max_second)
            self.population.append([first, second])
        

    def test(self):

        pool = mp.Pool(processes=os.cpu_count())

        for model_name in self.pop_stats:
            
            pool.apply_async(
                    model_trainer.test, 
                    args=(
                            model_name,
                            self.pop_stats,
                        )
                    )
            
        pool.close()
        pool.join()






    def train(self, gen: int):

        pool = mp.Pool(processes=os.cpu_count())

        for i in range(len(self.population)):
            #model_trainer.simulate(0.01, f"test_{i}", 100, self.population[i])

            self.pop_stats[f"{gen}_{i}"] = [self.population[i]]

            pool.apply_async(
                    model_trainer.simulate, 
                    args=(
                            0.01, 
                            f"test_{gen}_{i}", 
                            100, 
                            self.population[i],
                        )
                    )
            
        pool.close()
        pool.join()


    def evolve(self):

        self.train(0)
        return


def main():
    
    evolver = Evolution(515, 370)
    evolver.evolve()


if __name__ == '__main__':
    main()