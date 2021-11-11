# DEEP Q-VALUE REINFORCEMENT DAY TRADING

# Introduction
This git repo represents a personal project I've worked on the past four months and am continuing to work on for grad school projects. I currently work as a machine learning engineer for Copia Wealth Studios, a financial analysis company for ultra high net-worth clients. The majority of my work thus far has been building a production ML pipeline for document ingestion and ananlysis, with smaller projects focused on specific assest class clustering algorithims. 

There are currently two branches to this project. The animal named folders in src and crypto_1. The animal folders pertain to different veresions of a deep q-value reinforcement learning algorithim for short-term trading of the SPY stock. To acomplish this, the model uses two nested LSTM neural networks that take into consideration various price and volume factors for the SPY, VXUS, BND and VTI stocks. Furthermore, I have built a webscraping application to pull reddit articals pertaining to the general market. These articals are fed through an NLP sentiment analysis model and the correspodning sentiment score is then used as a stacked model input for real time market adjustments.

The crypto folder pertains to an extension of this idea, but to 14 different crypto coins. Each coin has has had a corresponding XGB boosted forecasting model trained on a target value of 4 weeks from the present minute. This forecasted prediction and the reddit sentiment analysis are joined into a stacked model with the nested LSTM networks for the internal reinforcement learning algorithim. This process is repeated for each of the 14 coins. The resulting reinforcement models are then run in parralel and fed into a ensamble SVG classifier model for general portfolio management, optimized on minimizing draw-down and maximizing general gains.

Finally, I have also developed a novel evolutionary optimizer for neural architecture. The code is working and optimized in parallel; however, I have yet to fully run it due to large compute times necessary for such an optimization.

Please refrence Road Map for the current development and status of the project. 

# Installation
 Firstly, you must install all required dependencies. To do this, please run:

 
    pip3 -r requirments.txt
 

Secondly, you must set your enviromental variables for access to the ALPACA api. To do this, simply run
the following commands: 


    export APCA_API_KEY_ID="<YOUR_KEY_HERE>"

    export APCA_API_SECRET_KEY="<YOUR_SECRET_KEY_HERE>"


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

    
    python3 model_trainer.py
    

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

    
    python3 live_enviroment.py
    

This will start a continually running script that will pull data from the ALPACA websocket api and make live trades
in real time using my paper trading account. The model will output the data being recieved each minute and the correspodning 
decision it makes.

Please note the data being recieved and the decision made are not synced together. As the websocket is called multiple times 
for each minute depending on order of recieving data, we cannot base decisions off this. Rather, the minute updates are handled by a separetly running function in parralel on a different thread. 

The script will do nothing when the market is not open, as it not recieving any data. It will also do nothing for the first 10 min of market open, as it needs to collect the first 10 min of history for model usage. 

Finally, please note that the model trainer and live model are sometimes disconnected. I am continual development and testing
of the model trainer. This may include trying new inputs or restructing the model architecture itself. To use the model in a live enviroment, the underlying architecture needs to match the model input space needs. As it would be poor development practices to continually change the live model trader when I am still developing the model itself, it is likley that you can not use the model you trained in the live model trainer as it will be mutliple iterations head. Thus, please use the prived model_850 for live model trading use. 

# Version Analysis

Please see this document for a detailed explanation on each enviroment and model

    https://docs.google.com/document/d/1lvRM6XtqxV6H4eJls4FsWzJUNFcDfPCCc9m46HDDoeo/edit?usp=sharing 

Please refrence this document for model training and performance tracking
    https://docs.google.com/document/d/1bdCvDyaqvKDXc7C1p2EyDhswiw93gYrBEnMIBK_-0oQ/edit?usp=sharing


Please view the following document for a comprehesnive test list and analysis of past configurations and performances of the model:
```
https://docs.google.com/document/d/1bdCvDyaqvKDXc7C1p2EyDhswiw93gYrBEnMIBK_-0oQ/edit?usp=sharing
```

# Contributing
This is a private project and a private repository. No external contributions are currently wanted or accepted. 

# Project Status
Currently in active development

# License
[Gilbert.Inc](https://choosealicense.com/licenses/agpl-3.0/)
