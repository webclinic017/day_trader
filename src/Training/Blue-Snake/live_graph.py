import matplotlib.pyplot as plt
import os
import pickle
import time

def main():

    ver = "60d"
    x = []
    monies = []

    plt.ion()
    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    
    line1, = ax.plot(x, monies)
    
    # setting title
    plt.title("Live Value Tracking", fontsize=20)
    
    # setting x-axis label and y-axis label
    plt.xlabel("Iteration")
    plt.ylabel("Value in USD")

    while(True):

        if os.path.isfile(f"models/{ver}/monies.pkl"):
            
            with open(f"models/{ver}/monies.pkl", "rb") as f:
                monies = pickle.load(f)

            x = list(range(len(monies)))

            line1.set_xdata(x)
            line1.set_ydata(monies)
        
            figure.canvas.draw()

            figure.canvas.flush_events()

        time.sleep(1)
        
    

if __name__ == '__main__':
    main()