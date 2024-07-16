# Library import
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import wget
import argparse
import importlib

# my_module을 수정한 후 다시 로드
import preprocess
importlib.reload(preprocess)
from preprocess import DataLoader
import visualization


def argsparsing():
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='Process some paths and model names.')
    
    # 명령줄 인수 추가
    
    parser.add_argument('--data', type=str, default='data', help='Dataset path')
    parser.add_argument('--target_data', type=str, default='dmv', help='Target dataset')    
    parser.add_argument('--wordset', type=str, default='wordset', help='Dataset path')    
    parser.add_argument('--output', type=str, default='output', help='Dataset path')
    parser.add_argument('--regular_stopword', type=str, default='stopword.csv', help='Stopword list')
    parser.add_argument('--multi_words', type=str, default='multi_words.csv', help='Multi wor list')
    parser.add_argument('--maufact_stopword', type=str, default='manufact_stopword.csv', help='AV manufacturer sentences')
    parser.add_argument('--keyword_dictionary', type=str, default='keyword.csv', help='Keyword dictionary')
    parser.add_argument('--replacement', type=str, default='replacement.csv', help='Word replacement dicionary')
    parser.add_argument('--auto_labeling_model', type=str, default='bert', help='Type of topic model')    
    parser.add_argument('--topic_model', type=str, default='lda', help='Type of auto-labeling model')
    parser.add_argument('--training', type=bool, default=True, help='Auto lageling mdoel training')
        
    args = parser.parse_args()
    
    print("=============================================")
    DATA_PATH = args.data    
    print(f"DATA_PATH: {DATA_PATH}")
    TARGET_DATA = args.target_data    
    print(f"TARGET_DATA: {TARGET_DATA}")
    FINAL_DATASET_PATH = os.path.join(os.getcwd(), DATA_PATH, TARGET_DATA)
    FINAL_DATASET_PATH += '/dmv.csv'
    
    print(f"FINAL_DATASET: {FINAL_DATASET_PATH}")
    if TARGET_DATA == 'dmv':
        if not os.path.exists(os.path.join(FINAL_DATASET_PATH)):
            print(f"{os.path.join(FINAL_DATASET_PATH)} does not exists")
            print("Downloading dmv2023 dataset...")
            url = 'https://www.dmv.ca.gov/portal/file/2023-autonomous-vehicle-disengagement-reports-csv/'
            wget.download(url, out=os.path.join(FINAL_DATASET_PATH))
            print(f"\n dmv2023 dataset downloaded successfully!")
            
            print(f"FINAL DATA PATH and NAME: {FINAL_DATASET_PATH}")
    
    else:
        print("Does not support other dataset yet")
        print("Program exit")
        return 0
    
    OUTPUT_PATH = os.path.join(os.getcwd(), args.output)
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")
    
    WORDSET_PATH = os.path.join(os.getcwd(), DATA_PATH, args.wordset)
    print(f"WORDSET: {WORDSET_PATH}")
    
    REGULAR_STOPWORD_FILE = args.regular_stopword  
    REGULAR_STOPWORD_FILE = os.path.join(WORDSET_PATH, REGULAR_STOPWORD_FILE)
    print(f"REGULAR_STOPWORD: {REGULAR_STOPWORD_FILE}")
    
    MANUFACT_STOPWORD_FILE = args.maufact_stopword       
    MANUFACT_STOPWORD_FILE = os.path.join(WORDSET_PATH, MANUFACT_STOPWORD_FILE)
    print(f"MANUFACT_STOPWORD: {MANUFACT_STOPWORD_FILE}")
    
    MULTI_WORDS_FILE = args.multi_words   
    MULTI_WORDS_FILE = os.path.join(WORDSET_PATH, MULTI_WORDS_FILE)
    print(f"MULTI_WORDS: {MULTI_WORDS_FILE}")
    
    REPLACEMENT_FILE = args.replacement   
    REPLACEMENT_FILE = os.path.join(WORDSET_PATH, REPLACEMENT_FILE)
    print(f"REPLACEMENT: {REPLACEMENT_FILE}")
    
    KEYWORD_DICT_FILE = args.keyword_dictionary
    KEYWORD_DICT_FILE = os.path.join(WORDSET_PATH, KEYWORD_DICT_FILE)    
    print(f"KEYWORD_DICT: {KEYWORD_DICT_FILE}")
    
    print("=============================================\n")
    
    return FINAL_DATASET_PATH, OUTPUT_PATH, MANUFACT_STOPWORD_FILE, REGULAR_STOPWORD_FILE, MULTI_WORDS_FILE, REPLACEMENT_FILE
    
    
def main():
    FINAL_DATASET_PATH, OUTPUT_PATH, MANUFACT_STOPWORD_FILE, REGULAR_STOPWORD_FILE, MULTI_WORDS_FILE, REPLACEMENT_FILE = argsparsing()
    daraloader = DataLoader(FINAL_DATASET_PATH, OUTPUT_PATH, MANUFACT_STOPWORD_FILE, REGULAR_STOPWORD_FILE, MULTI_WORDS_FILE, REPLACEMENT_FILE)
    
    daraloader.load_data()
    daraloader.preprocessing()
    
    
if __name__ == "__main__":
    main()