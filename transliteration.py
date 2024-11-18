from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import sys

if os.path.exists("/.dockerenv"):
    print("This file can't be run in a docker environment")
    sys.exit()

# Initialize Tkinter window (it will not be shown)
Tk().withdraw()

# to transliterate from devanagari scrpit to roman without translation
# Ask the user to select the file
file_path = askopenfilename(title="Select Dataset for training the model", filetypes=[("CSV files", "*.csv")])

# Load the dataset
if file_path:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
else:
    print("No file selected!")
    sys.exit()

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
