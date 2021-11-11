import requests
import pandas as pd
from datetime import datetime
import praw
import json
import time
from tqdm import tqdm
import flair
import multiprocessing as mp
import os
import pickle
import numpy as np


def comp_sent(data: list, shared_list: dict, i: int):
    
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    temp = []
    
    for k in tqdm(range(len(data))):
       
        s = flair.data.Sentence(str(data[k]))
        flair_sentiment.predict(s)
        total_sentiment = s.labels

        if total_sentiment:

            if "NEGATIVE" in str(total_sentiment):
                sentiment = -float(str(total_sentiment[0]).split()[-1].replace("(", "").replace(")", ""))
            elif "POSITIVE" in str(total_sentiment):
                sentiment = float(str(total_sentiment[0]).split()[-1].replace("(", "").replace(")", ""))
            else:
                sentiment = 0
        
        else:
            sentiment = 0

        
        temp.append(sentiment)
    
    shared_list[i] = temp
    print(temp)
    print(shared_list[i])
    with open(f"sentiment/{i}.pkl", 'wb') as f:
        pickle.dump(shared_list[i], f)


class ReditScraper():

    headers: dict
    reddit: praw.Reddit

    def __init__(self) -> None:
        
        #self.connect()
        #self.scrape()
        self.group_by_hour()

    def group_by_hour(self):
        
        df = pd.read_csv("sentiment.csv")
        df["created_time"] = pd.to_datetime(df["created_time"])

        master = {}
        
        for index, row in df.iterrows():

            date = datetime(row["created_time"].year, row["created_time"].month, row["created_time"].day, row["created_time"].hour)
            print(f"{date}: {row['sentiment']}")
            
            if(date not in master):
                master[date] = []
            
            master[date].append(float(row["sentiment"]))
        
        for key in master:
            master[key] = np.mean(master[key])
            print(f"{key}: {master[key]}")
        
        with open("hour_to_sentiment.pkl", "wb") as f:
            pickle.dump(master, f)



        

       

    def sentiment_analysis(self):

        df = pd.read_csv("a.csv")

        text_list = list(df["selftext"])
        inc_size = int(len(text_list)/100)

        pool = mp.Pool(os.cpu_count())
        shared_list = mp.Manager().list()

        for i in range(100):
            
            start = i * inc_size
            end = (i+1) * inc_size
            shared_list.append([])

            if i == 99:
                pool.apply_async(func=comp_sent, args=(text_list[start:], shared_list, i,))
            else:
                pool.apply_async(func=comp_sent, args=(text_list[start:end], shared_list, i,))
        
        pool.close()
        pool.join()

        combined_sent =[]

        for i in range(len(shared_list)):
            for sentiment in shared_list[i]:
                combined_sent.append(sentiment)
        

        df["sentiment"] = pd.Series(combined_sent)

        df.to_csv("sentiment.csv")


        
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