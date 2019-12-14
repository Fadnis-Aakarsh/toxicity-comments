#Toxicity Comments

#Abstract
The goal of our project is to automate the detection and flagging of toxic comments posted by users
on various online platforms. Through machine learning, we aim to build a robust toxic comment
detection system not reliant on a filter list. To make the technology usable, we plan to develop a
service/web application which moderators can use to flag or remove toxic comments. Thus far, we
have implemented a baseline model, trained it and analysed its performance on a preliminary dataset.
We have also created our final dataset and developed a plan of action for our next steps. All this
information is described further in detail below.

#Model
For the model, we trained the model on our collated Reddit comment dataset after doing
extraction and preprocessing.
The following parameter values were selected for tuning: -
• Maximum Sequence Length: 64,128
• Number of epochs: 8,16
• Training Batch Size: 1,2,3

This was integrated with web application for Reddit moderators to use. The online portal pull the 1000 newest comments that had been predicted as toxic by our trained model of the provided subreddit. The comments are stored in Database by comment collector using the PRAW API at regular intervals, and update them to a DB. All outstanding comments would then be processed at regular intervals by a REST API call to another node exposing our trained BERT model and then updated to the DB.
Our front end, written in React.JS , would then fetch the latest comments classified toxic for a specified subreddit from the database.



