# download_bert_model.py - Downloads and caches required models before training

# imports
import os
from transformers import AutoTokenizer, AutoModel


# define save directory for BERT model (will be saved in a 'pretrained_models/bert-base-uncased' folder relative to this script)
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'bert-base-uncased')

print(f'Downloading bert-base-uncased to {SAVE_DIR}...', flush=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# load and save tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print('Done. BERT model saved successfully.')




