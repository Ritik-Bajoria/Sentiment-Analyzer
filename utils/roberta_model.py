from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL ="cardiffnlp/twitter-roberta-base-sentiment"
TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, token=TOKEN)

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

    