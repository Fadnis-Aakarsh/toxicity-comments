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

    '''
    humansbeingbros
    mademesmile
    getmotivated
    wholesomememes
    randomkindness
    decidingtobebetter
    happycryingdads
    bettereveryloop
    askscience
    '''

    #slightly unsure
    '''
    keto
    intermittentfasting
    fasting
    '''

    subreddits = ['AskHistorians']#,'toastme','science','happy','aww','toptalent','upliftingnews','tipofmytongue','educationalgifs', 'gifsthatkeepongiving']
    outputfile = open('outputfile.csv','a+')
    all_comments = []
    total_comments_written = 0

    for cur_subreddit in subreddits:
        subreddit = redditClient.subreddit(cur_subreddit)

        top25posts = subreddit.top(params={'t': 'all'}, limit=25)
        
        for submission in top25posts:

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

            #write to disk if total length >=1000
            if len(all_comments) >= 1000:
                writer = csv.writer(outputfile, delimiter='\t')
                for comment in all_comments:
                    writer.writerow([comment.body])
                print("Wrote {0} comments".format(len(all_comments)))
                total_comments_written += len(all_comments)
                all_comments = []

        print("Done with subreddit {0}".format(cur_subreddit))

    #write remaining
    if len(all_comments) >= 0:
        writer = csv.writer(outputfile, delimiter='\t')
        for comment in all_comments:
            writer.writerow([comment.body])
        print("Wrote {0} comments".format(len(all_comments)))
        total_comments_written += len(all_comments)

    print("Total comments written {0}".format(total_comments_written))


