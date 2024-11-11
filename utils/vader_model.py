from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon (if needed)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# VADER Sentiment Analysis Function
def vader_sentiment(review):
    score = sia.polarity_scores(review)
    if score['compound'] >= 0.05:
        return "positive"
    elif score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"
