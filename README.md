# DEEP Q-VALUE REINFORCEMENT DAY TRADING

# Introduction
This git repo represents a personal project I've worked on the past four months and am continuing to work on for grad school projects. I currently work as a machine learning engineer for Copia Wealth Studios, a financial analysis company for ultra high net-worth clients. The majority of my work thus far has been building a production ML pipeline for document ingestion and ananlysis, with smaller projects focused on specific assest class clustering algorithims. 

I first began this project as I've also held a deep intrest in machine learning's application twords the stock market. It seems such a natural application of the concept. However, after any research one can easily see that this is everyone's first instinct, and it has not been widly accomplished due to one singular reason. There is no predictve correlation in historic stock price to it's current price. It's mathematically proven that a stock price, in relation to itself, is purley random. Thus, time-series analysis applied directly to this problem is a futile task. However, one must consdier that the individual stocks price is only a small subset of data one can gather when trading. For example, a day trader might have 15 different graphs open at once when analysing the market. With this in mind, we can approach the problem of machine learning's application to day trading with new vigour. If we can provide outside indicators to the problem, then surely we can train a model? Yet, most ML models are poorly suited for multi-factored numeric input over a time-series, and it would be impossible to create a training set of all possible states of the market for a typical classifiaction model.

This is where reinforcement learning comes. Allowing the algorithm to simply play in a market enviroment over n solutions and train itself seems the perfect solution. 

# Installation
 Firstly, you must install all required dependencies. To do this, please run:

 ```
 pip3 -r requirments.txt
 ```

Secondly, you must set your enviromental variables for access to the ALPACA api. To do this, simply run
the following commands: 

```
export APCA_API_KEY_ID="<YOUR_KEY_HERE>"

export APCA_API_SECRET_KEY="<YOUR_SECRET_KEY_HERE>"
```

As this is initially for a grad project, here are my personal keys to save whoever is grading this 
from having to create their own account: 

    Secret Key: uS3A2pY5vnnaLlDLeE46Yw2s3vBBpVE2Jn4M2Z1u
    Key: PK39Y29GBYTVVCWIPQO0

I understand the inate security risk of this; however my hope is that this is negated as this is a
private repository and the only people with access are my professors. So, as long as my proffesors don't
try and steal my CC info, I think we're set ;)


# Usage

There are two potential usages for this script. Firstly, you can decide to train the model using a combination
of both the model_trainer and train_enviroment script. Secondly, you can used your trained model to live trade
on a paper trading account through ALPACA.

Model Training:
    
To train your model simply run:

    ```
    python3 model_trainer.py
    ```

This will currently run for 1000 train cycles. Each training cycle is of abitrary length due to the reasons
outlined in the description. I've found it usually takes 36-48h to fully run on my macbook pro with 32gb of ram 
and a 2.3 GHz intel i7 processor. 

The script will save the state of the training for each iteration. So for every iteration completed, the script
will create the following files: target_model_it, model_it, X_it.pkl, Y_it.pkl, max_profits_it.pkl and replay_mem_it.pkl.
Where it is the current iteration of the training. All files will be saved under the cache folder.  

The progress of the training script can be seen in the terminal output through the use of tqdm, which will output a 
progress bar. The final money held by the model and the max money made by the model for the training cycle will
also be outputted for each iteration.

Live Model Trading:

To live trade with the model, simply run:

    ```
    python3 live_enviroment.py
    ```

This will start a continually running script that will pull data from the ALPACA websocket api and make live trades
in real time using my paper trading account. The model will output the data being recieved each minute and the correspodning 
decision it makes.

Please note the data being recieved and the decision made are not synced together. As the websocket is called multiple times 
for each minute depending on order of recieving data, we cannot base decisions off this. Rather, the minute updates are handled by a separetly running function in parralel on a different thread. 

The script will do nothing when the market is not open, as it not recieving any data. It will also do nothing for the first 10 min of market open, as it needs to collect the first 10 min of history for model usage. 

Finally, please note that the model trainer and live model are sometimes disconnected. I am continual development and testing
of the model trainer. This may include trying new inputs or restructing the model architecture itself. To use the model in a live enviroment, the underlying architecture needs to match the model input space needs. As it would be poor development practices to continually change the live model trader when I am still developing the model itself, it is likley that you can not use the model you trained in the live model trainer as it will be mutliple iterations head. Thus, please use the prived model_850 for live model trading use. 


# Road Map

# Contributing

# Project Status

# License