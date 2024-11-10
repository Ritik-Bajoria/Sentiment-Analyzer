from utils.logger import Logger
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import nltk

logger = Logger()

plt.style.use('ggplot')
app = Flask(__name__)


# Middleware: before each request
@app.before_request
def before_request_func():
    # This will run before every request
    logger.info(f"Request from {request.remote_addr} at {request.method} {request.url}")

@app.route('/api/sentiment-analysis',methods = ['GET'])
def sentimentAnalyzer():
    df = pd.read_csv('./Database/IMDB Dataset.csv')








if __name__ == '__main__':
    port = 4000
    app.run(debug=True, port=port)
    logger.info(f"Server started listening at port {port}")
