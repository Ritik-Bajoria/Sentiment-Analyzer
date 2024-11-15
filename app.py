#import all the necessary packages
from utils.logger import Logger
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import signal
from utils.vader_model import vader_sentiment
from utils.roberta_model import roberta_sentiment, roberta_sentiment_2

#create an instance of logger
logger = Logger()

#create a flask app
app = Flask(__name__)
CORS(app)


# Middleware: before each request
@app.before_request
def before_request_func():
    # This will run before every request
    logger.info(f"Request from {request.remote_addr} at {request.method} {request.url}")

#Routes for our API
@app.route('/api/sentiment-analysis',methods = ['GET'])
def sentimentAnalyzer():

    # importing the database
    df = pd.read_csv('./Database/IMDB Dataset.csv')

    # slicing the database into 500 rows
    # df = df.head(500)

    # converting all reviews to lowercase
    # df['text'] = df['text'].str.lower()
    # rob_score = 0
    # vad_score = 0
    # rob_2_score = 0
    # for i in range(0,5000):
    #     print(i)
    #     # taking one review from the database
    #     example = df['text'][i]
    #     # the original sentiment of the review
    #     org = df['sentiment'][i]
    #     # doing sentiment analysis using vader model
    #     vad = vader_sentiment(example)

    #     # doing sentiment analysis using roberta model
    #     rob = roberta_sentiment(example)
    #     rob_2 = roberta_sentiment_2(example)

    #     if vad == org:
    #         vad_score += 1

    #     if rob == org:
    #         rob_score +=1
        
    #     if rob_2 == org:
    #         rob_2_score += 1
    


    # # returning the results from various models in json format
    # return {
    #     "vader": vad_score,  
    #     "roberta": rob_score, 
    #     "my_roberta":rob_2_score,                  
    #     "original": i
    # }, 200

    example = "what a nice product"
    # doing sentiment analysis using vader model
    vad = vader_sentiment(example)
    # doing sentiment analysis using roberta model
    rob = roberta_sentiment(example)
    rob_2 = roberta_sentiment_2(example)
    # returning the results from various models in json format
    return {
        "vader": vad,  
        "roberta": rob, 
        "my_roberta":rob_2,                  
        "original": 'negative'
    }, 200

# Graceful shutdown function
def graceful_shutdown(signal, frame):
    logger.info("Shutting down gracefully...")
    # Perform any cleanup here if needed
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)



if __name__ == '__main__':
    port = 4000
    app.run(debug=True,host="127.0.0.1", port=port)

