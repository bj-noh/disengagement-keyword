import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import chardet
import os

# Create PyTorch datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings['input_ids'])

class AutoClassModel:
    def __init__(self, sample_labeled_file_path, non_labeled_file_path, output_path):
        print("Init Auto-Classification Model - BERT")
        self.sample_labeled_file_path = sample_labeled_file_path
        self.non_labeled_file_path = non_labeled_file_path
        self.output_path = output_path
        
    def __check_file_encoding(self, filename):             
        with open(filename, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        return encoding
    
    def __convert_to_utf8(self, file_path):
        # checking file encoding
        encoding = self.__check_file_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        
        directory, filename = os.path.split(file_path)        
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}-utf8{ext}"
        output_path = os.path.join(directory, new_filename)

        # return utf-8 encoded fileanme if file does not encode as UTF-8
        if encoding.lower() != 'utf-8':        
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"File converted to UTF-8 and saved as {output_path}")            
            return output_path
        
        else:
            print("File is already in UTF-8 encoding.")            
            return file_path

    def DataLoader(self):        
        print("Data files loading for auto-classification model training...")        
        self.sample_labeled_file_path = self.__convert_to_utf8(self.sample_labeled_file_path)
        self.sample_labeled_dat = pd.read_csv(self.sample_labeled_file_path, encoding='utf-8')
        # print("====================================")
        # print(self.sample_labeled_dat)

        self.non_labeled_file_path = self.__convert_to_utf8(self.non_labeled_file_path)
        self.non_labeled_dat = pd.read_csv(self.non_labeled_file_path, encoding='utf-8')
        # print("====================================")
        # print(self.non_labeled_dat)

        labeled_dat_s = self.sample_labeled_dat.drop_duplicates(subset=['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'])
        merged_dat = pd.merge(self.non_labeled_dat, labeled_dat_s[['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT', 'Label', 'Sub-label', 'sub-sub-label']], on='DESCRIPTION OF FACTS CAUSING DISENGAGEMENT', how='left', suffixes=('', '_B'))
        # origin_dat.update(merged_dat[['Label_B', 'Sub-label_B', 'sub-sub-label_B']].rename(columns={'Label_B': 'Label', 'Sub-label_B': 'Sub-label', 'sub-sub-label_B': 'sub-sub-label'}))
        self.non_labeled_dat['Label'] = merged_dat['Label_B']
        self.non_labeled_dat['Sub-label'] = merged_dat['Sub-label_B']
        self.non_labeled_dat['sub-sub-label'] = merged_dat['sub-sub-label_B']
        # origin_dat.to_csv('./data/final_labeled_data.csv', encoding='utf-8')

        self.non_matched_dat = self.non_labeled_dat[self.non_labeled_dat['Label'].isna()]
        self.matched_dat = self.non_labeled_dat[~self.non_labeled_dat['Label'].isna()]

        new_filename = 'ac_training.csv'
        output_path = os.path.join(self.output_path, new_filename)
        self.matched_dat.to_csv(output_path, encoding='utf-8')
        print(f"Preprocssed file is saved in {output_path}")

        new_filename = 'ac_test.csv'
        output_path = os.path.join(self.output_path, new_filename)
        self.non_matched_dat.to_csv(output_path, encoding='utf-8')
        print(f"Preprocssed file is saved in {output_path}")

    def train(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {self.device}')

        # Combine 'Label' and 'sub-label' into a single label for multi-class classification
        self.matched_dat['combined_label'] = self.matched_dat['Label'].astype(str) + "_" + self.matched_dat['Sub-label'].astype(str) + "_" + self.matched_dat['sub-sub-label'].astype(str)

        # Encode the labels
        label_encoder = LabelEncoder()
        self.matched_dat['encoded_label'] = label_encoder.fit_transform(self.matched_dat['combined_label'])

        # Split the labeled data into train and test sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            # labeled_data['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'], 
            self.matched_dat['PREP1_MANUFACT_STOPWORD_REMOVAL'], 
            self.matched_dat['encoded_label'], 
            test_size=0.2, 
            random_state=42
        )

        # Tokenization
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')

        def tokenize_function(texts):
            return tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")

        train_encodings = tokenize_function(train_texts.tolist())
        test_encodings = tokenize_function(test_texts.tolist())
        # non_labeled_encodings = tokenize_function(non_labeled_data['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'].tolist())
        non_labeled_encodings = tokenize_function(self.non_labeled_dat['PREP1_MANUFACT_STOPWORD_REMOVAL'].tolist())

        train_dataset = CustomDataset(train_encodings, train_labels.tolist())
        test_dataset = CustomDataset(test_encodings, test_labels.tolist())
        non_labeled_dataset = CustomDataset(non_labeled_encodings)

        # Fine-tune BERT
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
        # model = ElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator', num_labels=len(label_encoder.classes_))
        print(model)

        training_args = TrainingArguments(
            output_dir='./results',          
            num_train_epochs=30,              
            per_device_train_batch_size=4,  # Reduce batch size
            per_device_eval_batch_size=4,   # Reduce eval batch size
            gradient_accumulation_steps=2,  # Use gradient accumulation
            fp16=True,                      # Enable mixed precision training
            warmup_steps=500,                
            weight_decay=0.01,               
            logging_dir='./logs',            
            logging_steps=100,                # Log every 100 steps to see progress
            evaluation_strategy="epoch"
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            precision = precision_score(labels, preds, average='weighted')
            recall = recall_score(labels, preds, average='weighted')
            return {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                # 'f1': f1,
            }

        trainer = Trainer(
            model=model,                        
            args=training_args,                  
            train_dataset=train_dataset,         
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics          
        )

        trainer.train()

        # Predict labels for non-labeled data
        predictions = trainer.predict(non_labeled_dataset)
        predicted_labels = predictions.predictions.argmax(axis=1)

        # Decode the labels
        decoded_labels = label_encoder.inverse_transform(predicted_labels)

        # Split the combined labels back into 'Label' and 'sub-label'
        self.non_labeled_dat['combined_label'] = decoded_labels
        self.non_labeled_dat[['Label', 'Sub-label', 'sub-sub-label']] = self.non_labeled_dat['combined_label'].str.split('_', expand=True)

        self.non_labeled_dat.to_csv('./output/result.csv')

        trainer.evaluate()