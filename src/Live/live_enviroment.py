from datetime import datetime
import alpaca_trade_api as tradeapi
import websocket, json
from collections import deque
import os
from live_model import LiveModel
from threading import Thread
import numpy as np
from datetime import date, timedelta
from dateutil import parser
import logging
import ecs_logging
import ast
import sys


class LiveEnviroment():

    logger: logging.Logger
    host = 'localhost'
    port = 5000


    api: tradeapi.REST
    auth_data: dict
    socket_url: str
    ws: websocket
    prev_min: datetime

    live_model: LiveModel

    def __init__(self, it: int) -> None:
        """ Base level initializer for the class. 
        Creates new API, connects to the API websocket
        and begins running the socket connection
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """

        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('day_trader_logging.json')
        handler.setFormatter(ecs_logging.StdlibFormatter())
        self.logger.addHandler(handler)
        
        self.logger.info("Init", extra={"http.request.body.content": "Starting application"})

        self.api = tradeapi.REST()
        self.auth_data = {
            "action": "authenticate",
            "data": {"key_id": os.environ.get("APCA_API_KEY_ID", None), "secret_key": os.environ.get("APCA_API_SECRET_KEY", None)}
        }
        self.socket_url = "wss://data.alpaca.markets/stream"
        self.ws = websocket.WebSocketApp(self.socket_url , on_open=self.run, on_message=self.update_prices, on_close=self.close)

        self.live_model = LiveModel(it)

        self.ws.run_forever()

    def run(self, ws: websocket) -> None:
        """ Function called when websocket is initially called.
        Sends authentification data to websocket and starts background
        thread for action loop.
        
        Arguments:
            None

        Return:
           None
        
        Side Effects:
            Starts new thread running start_action_loop function
        """

        self.logger.info("run", extra={"http.request.body.content": f"Starting run function"})

        ws.send(json.dumps(self.auth_data))
        listen_message = {"action": "listen", "data": {"streams": ["AM.SPY", "AM.VXUS", "AM.VTI", "AM.BND"]}}
        ws.send(json.dumps(listen_message))

       
        thread = Thread(target = self.live_model.start_action_loop, args = ())
        thread.start()

    def close(self) -> None:
        """ Function called when websocket is closed.
        
        Arguments:
            None

        Return:
            None
        
        Side Effects:
            None
        """
        self.logger.info("close", extra={"http.request.body.content": f"Closing web socket"})
        print(f"Closed connection")
        return

    def update_prices(self, ws: websocket, message: dict) -> None:
        """ Function called when websocket recieves new information.
        Based on which stock is being updated, the corresponding price 
        data is added to the live_model's prev_10min deque. The price is 
        also normalized using log to match training data. If SPY is all the way full,
        then fill the rest with fill_rest function.
        
        Arguments:
            None

        Return:
            ws (websocket): The calling websocket

            message (dict): The dictionary json data send by the websocket
        
        Side Effects:
            None
        """


        if sys.getsizeof(message) < 10000:
            message = ast.literal_eval(message)
        
        else:
            self.logger.info("update_prices", extra={"http.request.body.content": f"Price message was too big, not evaluating"})
            return
            

        self.logger.info("update_prices", extra={"http.request.body.content": f"Got new price: {message}"})

        if message["stream"] == "AM.VTI":
            curr_VTI_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_VTI.append(curr_VTI_price)
            
        
        elif message["stream"] == "AM.VXUS":
            curr_VXUS_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_VXUS.append(curr_VXUS_price)
            
        
        elif message["stream"] == "AM.BND":
            curr_BND_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_BND.append(curr_BND_price)
            
        
        elif message["stream"] == "AM.SPY":
            curr_SPY_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_SPY.append(curr_SPY_price)

            if (len(self.live_model.prev_10_SPY) == 10 and len(self.live_model.prev_10_VXUS) > 0
                and len(self.live_model.prev_10_BND) > 0 and len(self.live_model.prev_10_VTI) > 0):
                self.live_model.fill_remaining()

                self.logger.info("update_prices", extra={"http.request.body.content": f"Reached 10 prices in spy and atleast one in the others, filling now"})



def main():
    model_live = LiveEnviroment(850)
    return

if __name__ == '__main__':
    main()