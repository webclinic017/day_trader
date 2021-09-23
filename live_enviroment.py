import alpaca_trade_api as tradeapi
import websocket, json
from collections import deque
import os
from live_model import LiveModel
from threading import Thread
import numpy as np

class LiveEnviroment():

    api: tradeapi.REST
    auth_data: dict
    socket_url: str
    ws: websocket

    live_model: LiveModel


    def __init__(self) -> None:
        
        self.api = tradeapi.REST()
        self.auth_data = {
            "action": "authenticate",
            "data": {"key_id": os.environ.get("APCA_API_KEY_ID", None), "secret_key": os.environ.get("APCA_API_SECRET_KEY", None)}
        }
        self.socket_url = "wss://data.alpaca.markets/stream"
        self.ws = websocket.WebSocketApp(self.socket_url , on_open=self.run, on_message=self.update_prices, on_close=self.close)

        self.live_model = LiveModel()

        self.ws.run_forever()

    def run(self, ws: websocket):

        ws.send(json.dumps(self.auth_data))
        listen_message = {"action": "listen", "data": {"streams": ["AM.SPY", "AM.VXUS", "AM.VTI", "AM.BND"]}}
        ws.send(json.dumps(listen_message))

        thread = Thread(target = self.live_model.start_action_loop, args = ())
        thread.start()

    def close(self):

        print(f"Closed connection")

    def update_prices(self, ws: websocket, message: dict):
        
        message = eval(message)             # Security risk - Fix later
        print(message)

        if message["stream"] == "AM.VTI":
            curr_VTI_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_VTI.append(curr_VTI_price)
            print(curr_VTI_price)
            
        
        elif message["stream"] == "AM.VXUS":
            curr_VXUS_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_VXUS.append(curr_VXUS_price)
            print(curr_VXUS_price)
            
        
        elif message["stream"] == "AM.BND":
            curr_BND_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_BND.append(curr_BND_price)
            print(curr_BND_price)
            
        
        elif message["stream"] == "AM.SPY":
            curr_SPY_price = round(np.log(message["data"]["c"]), 3)
            self.live_model.prev_10_SPY.append(curr_SPY_price)
            print(curr_SPY_price)

            if (len(self.live_model.prev_10_SPY) == 10 and len(self.live_model.prev_10_VXUS) > 0
                and len(self.live_model.prev_10_BND) > 0 and len(self.live_model.prev_10_VTI) > 0):
                self.live_model.fill_remaining()