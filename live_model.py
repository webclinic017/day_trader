import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import random
import alpaca_trade_api as tradeapi
from datetime import date, timedelta
import matplotlib.pyplot as plt
import pickle
import os.path
import os
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import shutil
import matplotlib.pyplot as plt
from dateutil import parser
from enviroment import Enviroment