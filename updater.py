import sys
import praw
import time
from datetime import datetime
import psycopg2
import requests
import threading
import json

params = {'host':'localhost','database':'toxicity','user':'toxic','password':'toxicity'}
redditClient = None
subreddits = None
lastUpdatedTime = 0.0
bertUrl = 'http://bert:5000/toxicity'

class MyThread (threading.Thread):
    die = False
    func = None
    def __init__(self, name, func, duration = 5*60):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.duration = duration

    def run (self):
        while not self.die:
            time.sleep(self.duration)
            self.func()

    def join(self):
        self.die = True
        super().join()

def processComments():

    #process all unprocessed comments
    sql = "select id, comment_data from reddit_comments where processing_status = 'unprocessed'"
    unprocComments = None

    conn = None
    try:
        conn= psycopg2.connect(**params)
        cur = conn.cursor()
        #get all unprocessed comments
        cur.execute(sql)
        unprocComments = cur.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error {0} occurred while fetching unprocessed comments to DB".format(error))
    finally:
        if conn is not None:
            conn.close()

    predictions = []
    if unprocComments is None or len(unprocComments)==0:
        return

    predictions = {}
    try:    
        for comment in unprocComments:
            response = None
            response = requests.get(bertUrl, params = { 'text': comment[1] })
            predictions[comment[0]] = (json.loads(response.text)['toxicity'])
    except Exception as e:
        print("Error {0} occurred while computing prediction for comment".format(e))

    sql = "update reddit_comments set processing_status='processed', prediction= %s, prediction_probability = '100.0' where id=%s"
    try:
        conn= psycopg2.connect(**params)
        cur = conn.cursor()
        #update comment status and predictions
        for key in predictions:
            cur.execute(sql, (predictions[key], key))
            conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error {0} occurred while updating processed comment statuses to DB".format(error))
    finally:
        if conn is not None:
            conn.close()

def fetchComments():
    global lastUpdatedTime    
    lock = threading.RLock()
    with lock:
        sr = subreddits

    sql ="""INSERT INTO reddit_comments (reddit_identifier, subreddit_name, comment_url, 
    comment_data, processing_status, user_action, prediction_probability, time, submitter_name, prediction, submitter_avatar) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    comments = []
    for subreddit in sr:
        #fetch latest 1000 comments    
        comments += [cmt for cmt in redditClient.subreddit(subreddit).comments(limit=1000) if cmt.created_utc > lastUpdatedTime]

    lastUpdatedTime = time.time()
    
    conn = None
    try:
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        #put in db
        for cmt in comments:
            cur.execute(sql, (cmt.id, cmt.subreddit.display_name, cmt.permalink, cmt.body, 'unprocessed', 'no_action', '0.0', datetime.utcfromtimestamp(cmt.created_utc), cmt.author.name, 'non_toxic', cmt.author.icon_img))
            conn.commit()
        cur.close()
        conn.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error {0} occurred while writing comments to DB".format(error))
    finally:
        if conn is not None:
            conn.close()

if __name__=='__main__':

    redditClient = praw.Reddit(client_id='8LNDKl7HusySlQ',
                    client_secret='bvNAk41kz5FW4bJvp_urp5prQbw',
                    password='throwawayTester_blah',
                    user_agent='throwawayTester_blah',
                    username='throwawayTester_blah')

    subreddits = [sys.argv[1]]
    f = MyThread('fetchComments', fetchComments)
    f.start()
    s = MyThread('processComments', processComments)
    s.start()
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        f.join()
        s.join()


"""

This module responsible for

##Backend activities

# fetch comments from reddit for a specified number of subreddits, add unprocessed to DB

# proess unprocessed add to db, in case of failure retry

"""