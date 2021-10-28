import requests
import pandas as pd
from datetime import datetime
import praw
import json
import time
from tqdm import tqdm
import os

class ReditScraper():

    headers: dict
    reddit: praw.Reddit

    def __init__(self) -> None:
        
        self.connect()
        self.scrape()

    def scrape(self):
        
        # Scrape for each hour and then combine the results together into one big df, just delete repeats
        df = pd.DataFrame()
        i = 1535241600 # 8/26/18 - 12am
        k = 1535245200 # 8/26/18 - 1am

        end = 1630022400 # 8/27/2021 - 12am
        a = 0

      
        while(k <= end):
            
            res = requests.get(f"https://api.pushshift.io/reddit/search/submission/?subreddit=wallstreetbets&before={k}&after={i}&size=100")
            
            try:
                res = json.loads(res.text)
            except:
                time.sleep(1)
                continue
            
                     
            for post in res["data"]:
                
                if "selftext" in post:
                    
                    df = df.append({
                        'title': post['title'],
                        'selftext': post['selftext'],
                        'id': post['id'],
                        'created_time': datetime.fromtimestamp(post['created_utc']),
                        'name': post["id"],
                        'hour': a
                    }, ignore_index=True)
                
                else:
                    print("no selftext")

            i += 3600
            k += 3600
            a += 1

            print(f"{a}/{int((end-k)/3600)}")

            if a > 99 and a % 100 == 0:
                df.to_csv("a.csv")


           
            
            

            




    def connect(self):

        # note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
        auth = requests.auth.HTTPBasicAuth('qMMTF0Ck5Aae-QPgZnVUeQ', 'yuIyd14kaPe2KAOQhDXe4ltRRtvcJw')

        # here we pass our login method (password), username, and password
        data = {'grant_type': 'password',
                'username': 'shanghai_mozzie',
                'password': 'Frankfurt22!'}

        # setup our header info, which gives reddit a brief description of our app
        headers = {'User-Agent': 'MyBot/0.0.1'}

        # send our request for an OAuth token
        res = requests.post('https://www.reddit.com/api/v1/access_token',
                            auth=auth, data=data, headers=headers)

        # convert response to JSON and pull access_token value
        TOKEN = res.json()['access_token']

        # add authorization to our headers dictionary
        self.headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

        # while the token is valid (~2 hours) we just add headers=headers to our requests
        requests.get('https://oauth.reddit.com/api/v1/me', headers=self.headers)

        self.reddit = praw.Reddit(client_id='qMMTF0Ck5Aae-QPgZnVUeQ', client_secret='yuIyd14kaPe2KAOQhDXe4ltRRtvcJw', user_agent='scraper')


    
        
    
def main():

    test = ReditScraper()
    

if __name__ == '__main__':
    main()