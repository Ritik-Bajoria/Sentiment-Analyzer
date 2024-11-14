from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import pandas as pd

# to transliterate from devanagari scrpit to roman without translation
df = pd.read_csv("./Database/collected_labeled_data.csv")
df = df.dropna()
reviews = df['text']
df_in_roman = {}
df_in_roman['text'] = []
df_in_roman['label'] = df['label']
for review in reviews:
    review_in_roman = transliterate(review, sanscript.DEVANAGARI, sanscript.ITRANS)
    df_in_roman['text'].append((review_in_roman.lower()))

df_in_roman = pd.DataFrame(df_in_roman)
df_in_roman.to_csv("./Database/Collected_labeled_nepali_data.csv",index=False)
