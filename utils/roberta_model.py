from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL ="cardiffnlp/twitter-roberta-base-sentiment"
TOKEN = "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"
tokenizer = AutoTokenizer.from_pretrained(MODEL,token = TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL,token = TOKEN)

MY_MODEL = "./roberta_finetuned_model"
my_tokenizer = AutoTokenizer.from_pretrained(MY_MODEL)
my_model = AutoModelForSequenceClassification.from_pretrained(MY_MODEL)

def roberta_sentiment(review):
    encoded_text = tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # encoded_text = tokenizer(review, return_tensors = 'pt')
    output = model(**encoded_text)
    score = output[0][0].detach().numpy()
    scores = softmax(score)
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }
    # Find the sentiment with the highest score
    predicted_sentiment = max(scores_dict, key=scores_dict.get)
    
    return predicted_sentiment  # Return the sentiment with the highest score

def roberta_sentiment_2(review):
    encoded_text = my_tokenizer(review, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # encoded_text = tokenizer(review, return_tensors = 'pt')
    output = my_model(**encoded_text)
    score = output[0][0].detach().numpy()
    scores = softmax(score)
    scores_dict = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }
    # Find the sentiment with the highest score
    predicted_sentiment = max(scores_dict, key=scores_dict.get)
    
    return predicted_sentiment  # Return the sentiment with the highest score
