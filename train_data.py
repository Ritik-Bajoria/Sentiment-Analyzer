from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import sys

if os.path.exists("/.dockerenv"):
    print("This file can't be run in a docker environment")
    sys.exit()

# Initialize Tkinter window (it will not be shown)
Tk().withdraw()

# Ask the user to select the file
file_path = askopenfilename(title="Select Dataset for training the model", filetypes=[("CSV files", "*.csv")])

# Load the dataset
if file_path:
    dataset_1 = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
else:
    print("No file selected!")
    sys.exit()

sentiment = input("enter the name of column containing sentiment:\t")
text = input("enter the name of column containing reviews:\t")
positive = input("enter categorical text used for positive:\t")
neutral = input("enter categorical text used for neutral:\t")
negative = input("enter categorical text used for negative:\t")

dataset_1['label'] = dataset_1[sentiment].map({positive: 2, neutral: 1, negative: 0})  # Adjust mapping if necessary

# Split the dataset into train and test sets
train_df, test_df = train_test_split(dataset_1, test_size=0.2)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict
dataset_1 = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

trained_model = './roberta_finetuned_model'

if os.path.exists(trained_model) and os.path.isdir(trained_model):
    model = RobertaForSequenceClassification.from_pretrained(trained_model)
    tokenizer = RobertaTokenizer.from_pretrained(trained_model)
else:
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples[text], padding=True, truncation=True, max_length=128)

tokenized_datasets = dataset_1.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Train the model
trainer.train()

# Save the model to a directory
model.save_pretrained('./roberta_finetuned_model')

# Save the tokenizer to the same directory
tokenizer.save_pretrained('./roberta_finetuned_model')