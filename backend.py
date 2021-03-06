#!/usr/bin/env python
from flask import Flask, request, Response
import jsonpickle
import sys
import psycopg2
import requests
import threading
import json
import io

# Initialize the Flask application
app = Flask(__name__)

params = {'host':'localhost','database':'toxicity','user':'toxic','password':'toxicity'}


@app.route('/update/action', methods=['POST'])
def update_db():

    r = request

    json_data = json.loads(io.BytesIO(r.data).read().decode('utf-8'))
    user_action = json_data['user_action']
    id = json_data['id']

    sql = """ UPDATE reddit_comments 
                    SET user_action = %s
                    WHERE reddit_identifier = %s"""


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
    except(Exception, psycopg2.DatabaseError) as e:
        print(e)
        error = 'Error {0} occurred while updating user action {1} for comment id {2}'.format(e,user_action,id)
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
    
    sql = """select * from reddit_comments where subreddit_name = %s and user_action = 'no_action' and  processing_status = 'processed' and prediction='toxic'
        order by time desc limit %s """

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
            objDict = {'reddit_identifier':cmt[1], 'subreddit_name':cmt[2], 'comment_url':'https://www.reddit.com'+cmt[3], 
            'comment_data':cmt[4], 'processing_status':cmt[5], 'user_action':cmt[6], 'prediction_probability':cmt[7], 'time':str(cmt[8]), 'submitter_name':cmt[9], 'prediction':cmt[10], 'submitter_avatar':cmt[11]}
            response['data'].append(objDict)
        response = jsonpickle.encode(response)
        return Response(response=response,status=200,mimetype='application/json')

# start flask app
if __name__=='__main__':
    app.run(host="0.0.0.0", port=5001)

"""

This module responsible for 

## APIs

# rest API fetch n toxic processed comments for a subreddit params number, subreddit

# update user action for a processed comment based on user selection

"""


## expose model 

# get prediction


## DBs

#Schema

#id reddit_identifier subreddit_name comment_url comment_data processing_status user_action prediction_probability prediction timestamp

## Frontend