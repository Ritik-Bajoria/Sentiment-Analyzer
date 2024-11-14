from utils.logger import Logger
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import signal
from utils.vader_model import vader_sentiment
from utils.roberta_model import roberta_sentiment, roberta_sentiment_2
logger = Logger()
app = Flask(__name__)
CORS(app)


# Middleware: before each request
@app.before_request
def before_request_func():
    # This will run before every request
    logger.info(f"Request from {request.remote_addr} at {request.method} {request.url}")

@app.route('/api/sentiment-analysis',methods = ['GET'])
def sentimentAnalyzer():
    # importing the database
    df = pd.read_csv('./Database/IMDB Dataset.csv')
    # slicing the database into 500 rows
    df = df.head(500)
    # converting all reviews to lowercase
    df['text'] = df['text'].str.lower()
    rob_score = 0
    vad_score = 0
    # taking one review from the database
    example = "If you enjoy questionable stains, paper-thin walls, and the feeling of rough sandpaper sheets, this is the place for you. It really takes the phrase 'roughing it' to a new level."
    # doing sentiment analysis using vader model
    vad = vader_sentiment(example)

    # doing sentiment analysis using roberta model
    rob = roberta_sentiment(example)
    rob_2 = roberta_sentiment_2(example)
    
   # the original sentiment of the review
    org = 'negative'

    # returning the results from various models in json format
    return {
        "vader": vad,  
        "roberta": rob, 
        "my_roberta":rob_2,                  
        "original": org
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

