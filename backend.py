#!/usr/bin/env python
from flask import Flask, request, Response
import jsonpickle
import sys
import praw
import time
from datetime import datetime
import psycopg2
import requests
import threading

# Initialize the Flask application
app = Flask(__name__)


params = {'host':'localhost','database':'toxicity','user':'toxic','password':'toxicity'}
redditClient = None
subreddits = None
lastUpdatedTime = 0.0
bertUrl = 'http://bert:5000/toxicity?text=%'

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
            response = requests.get(bertUrl % (comment[1]))
            predictions[comment[0]] = (json.loads(response)['toxicity'])
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


@app.route('/update/action', methods=['POST'])
def update_db():

    r = request

    json_data = json.loads(io.BytesIO(r.data).read().decode('utf-8'))
    user_action = json_data['user_action']
    id = json_data['id']

    sql = """ UPDATE reddit_comments 
                    SET user_action = %s
                    WHERE id = %s"""

    r= request
    conn = None
    error = None
    try:
        # check for database config
       	#hope this is defined
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(sql, (user_action, id))
        conn.commit()
        cur.close()
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
        error = 'Error {0} occurred while updating user action {1} for comment id {2}'.format(error,user_action,id)
    finally:
        if conn is not None:
            conn.close()

    if error is None:
        response = jsonpickle.encode({'response':'OK'})
        return Response(response=response, status=200, mimetype='application/json')
    else:
        response = jsonpickle.encode({'response':error})
        return Response(response=response,status=500,mimetype='application/json')

@app.route('/getcomments/<subreddit>/<count>', methods=['GET'])
def getProcessedComments(subreddit, count):

    #return processed comment information from the DB as json
    response = None
    
    sql = """select * from reddit_comments where subreddit_name = %s and user_action = 'no_action' and  processing_status = 'processed'
        limit %s order by time desc"""

    error = None
    conn = None
    res = []
    try:
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(sql, (subreddit,count))
        res = cur.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        error = 'Error {0} occurred while fetching processed comments for subreddit {1}'.format(error, subreddit)
    finally:
        if conn is not None:
            conn.close()
    
    #build response
    if error is not None:
        response = jsonpickle.encode({'status':error, 'data':None})
        return Response(response=response,status=500,mimetype='application/json')
    else:
        response = {}
        response['status'] = 'OK'
        response['data'] = []
        for cmt in res:
            #create dict
            objDict = {'reddit_identifier':cmt[0], 'subreddit_name':cmt[1], 'comment_url':cmt[2], 
            'comment_data':cmt[3], 'processing_status':cmt[4], 'user_action':cmt[5], 'prediction_probability':cmt[6], 'time':cmt[7], 'submitter_name':cmt[8], 'prediction':cmt[9], 'submitter_avatar':cmt[10]}
            response['data'].append(objDict)
        response = jsonpickle.encode(response)
        return Response(response=response,status=200,mimetype='application/json')

# start flask app
if __name__=='__main__':

    redditClient = praw.Reddit(client_id='8LNDKl7HusySlQ',
                    client_secret='bvNAk41kz5FW4bJvp_urp5prQbw',
                    password='throwawayTester_blah',
                    user_agent='throwawayTester_blah',
                    username='throwawayTester_blah')

    subreddits = sys.argv[1]
    timer1 = threading.Timer(5*60,fetchComments)
    timer2 = threading.Timer(5*60,processComments)

    app.run(host="0.0.0.0", port=5000)



##Backend activities

# fetch comments from reddit for a specified number of subreddits, add unprocessed to DB

# proess unprocessed add to db, in case of failure retry


## APIs

# rest API fetch n toxic processed comments for a subreddit params number, subreddit

# update user action for a processed comment based on user selection


## expose model 

# get prediction


## DBs

#Schema

#id reddit_identifier subreddit_name comment_url comment_data processing_status user_action prediction_probability prediction timestamp

## Frontend