import praw
import csv


if __name__ == '__main__':
    redditClient = praw.Reddit(client_id='8LNDKl7HusySlQ',
                     client_secret='bvNAk41kz5FW4bJvp_urp5prQbw',
                     password='throwawayTester_blah',
                     user_agent='throwawayTester_blah',
                     username='throwawayTester_blah')

    '''
    r/upliftingnews - neutral, good, slighlty positive
    r/tipofmytongue - neutral, 
    r/nostupidquestions- neutral
    r/coolguides - nuetral, slight hint of profanity
    r/Insighfulquestions - neutral, slight abusive language profanity, small traces of toxicity
    r/educationalgifs - neutral/slighlty positigve sentiment
    r/BetterEveryLoop - slightly positive sentiment on most of the top posts
    r/gifsthatkeepon giving - neutral, again, slightly +ve sentiment
    '''

    subreddits = ['AskHistorians','toastme','science','happy','aww','toptalent','upliftingnews','tipofmytongue','educationalgifs', 'gifsthatkeepongiving']
    all_comments = []

    for cur_subreddit in subreddits:
        subreddit = redditClient.subreddit(cur_subreddit)

        top25Posts = subreddit.top(params={'t': 'all'}, limit=25)

        
        for submission in top25Posts:

            #sort comments by top
            submission.comment_sort = 'top'
            
            #replace "morecomments"
            submission.comments.replace_more()

            #flatten list
            submission_comments = submission.comments.list()

            #get all comments with at least 100 points and add to all comments after replacing newlines with spaces
            submission_comments = filter(lambda comment: comment.body.find("&#x200B;") == -1 and comment.score>=100, submission_comments)

            for comment in submission_comments:
                comment.body = comment.body.replace('\n',' ')
                all_comments.append(comment)

    #put all comments in file
    with open('outputfile.csv', 'w') as f:
        writer = csv.writer(f,delimiter='\t')
        for comment in all_comments:
            writer.writerow([comment.body])

    print("Total comments {0}".format(len(all_comments)))


