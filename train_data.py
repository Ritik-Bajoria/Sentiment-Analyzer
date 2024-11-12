from transformers import RobertaForSequenceClassification, RobertaTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
dataset_1 = pd.read_csv("C:/Users/Legion/Ritik/Desktop/Programming/Intern work/04-Intern/Sentiment Analysis/Database/IMDB Dataset.csv")

# Ensure the label column is present and contains values (e.g., 0, 1, 2 for a 3-class classification problem)
# Assuming 'text' is the text column and 'label' is the label column in the IMDB dataset
dataset_1['label'] = dataset_1['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})  # Adjust mapping if necessary

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

# Now, load the model from the saved state to continue training on the second dataset
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=128)

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

model = RobertaForSequenceClassification.from_pretrained('./roberta_finetuned_model')
tokenizer = RobertaTokenizer.from_pretrained('./roberta_finetuned_model')

# Load your dataset
dataset_2 = pd.read_csv("C:/Users/Legion/Ritik/Desktop/Programming/Intern work/04-Intern/Sentiment Analysis/Database/IMDB Dataset.csv")

# Ensure the label column is present and contains values (e.g., 0, 1, 2 for a 3-class classification problem)
# Assuming 'text' is the text column and 'label' is the label column in the IMDB dataset
dataset_2['label'] = dataset_2['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})  # Adjust mapping if necessary

# Split the dataset into train and test sets
train_df, test_df = train_test_split(dataset_2, test_size=0.2)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_datasets = dataset_2.map(tokenize_function, batched=True)

# Create a DatasetDict
dataset_2 = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results_second',
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

# Train the model on the second dataset
trainer.train()

# Optionally, save the model after the second training
model.save_pretrained('./model_after_second_training')
tokenizer.save_pretrained('./model_after_second_training')