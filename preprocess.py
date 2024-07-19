import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
import chardet

class DataPreLoader:
    def __init__(self, FINAL_DATASET, OUTPUT_PATH, MANUFACT_STOPWORD, REGULAR_STOPWORD, MULTI_WORDS, REPLACEMENT):
        print("Init DataPreproc()...")
        
        print("Download nlp libary in nlp")
        nltk.download('stopwords')
        nltk.download('punkt')
        
        self.data_path = FINAL_DATASET
        self.stopword_path = REGULAR_STOPWORD
        self.manufact_stopword_path = MANUFACT_STOPWORD
        self.multi_words_path = MULTI_WORDS
        self.replacement_dict_path = REPLACEMENT
        self.output_path = OUTPUT_PATH

        nltk.download('stopwords')
        self.general_stopwords = stopwords.words('english')
        
        # self.replacement_dict = self.load_replacement_dict()
        # self.multi_word_terms = [
        #     'traffic light', 'autonomous mode', 'autonomous vehicle', 'unexpected behavior',
        #     'lane markings', 'traffic signal', 'safety reasons', 'stop sign', 'parking facility', 'ego vehicle',
        #     'increase velocity', 'reduce velocity', 'incorrect perception', 'lane change',
        # ]
        
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
        
        # print(output_path)
        
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
    
    
    # method for data (csv) and wordset loading
    def load_data(self):        
        print("Data files loading...")
        
        self.data_path = self.__convert_to_utf8(self.data_path)
        self.data = pd.read_csv(self.data_path, encoding='utf-8')
        
        self.stopword_path  = self.__convert_to_utf8(self.stopword_path)
        self.stopword = pd.read_csv(self.stopword_path, encoding='utf-8')
        
        self.manufact_stopword_path = self.__convert_to_utf8(self.manufact_stopword_path)
        self.manufact_stopword = pd.read_csv(self.manufact_stopword_path, encoding='utf-8')
        
        self.multi_words_path = self.__convert_to_utf8(self.multi_words_path)
        self.multi_words = pd.read_csv(self.multi_words_path, encoding='utf-8')
        
        self.replacement_dict_path = self.__convert_to_utf8(self.replacement_dict_path)        
        self.replacement_dict = pd.read_csv(self.replacement_dict_path, encoding='utf-8')
        
        return self.data, self.stopword, self.manufact_stopword, self.multi_words, self.replacement_dict
    
    def __remove_manufact_stopwords(self, text, stops):
        for stop in stops:
            text = text.replace(stop, '')        
        return text

    def __replace_multi_words(self, text, multi_word_terms):
        for term in multi_word_terms:
            text = text.replace(term, term.replace(' ', '_'))        
        return text

    def __remove_general_stopwords(self, text, additional_stopwords):
        
        final_stopwords = self.general_stopwords + additional_stopwords

        words = word_tokenize(re.sub(r'[^a-zA-Z\s_]', ' ', text.lower()))
        words = [word for word in words if word not in final_stopwords]
        
        return ' '.join(words)

    def __replace_general_form(self, text, replacement_dict):
        for other, key in replacement_dict.items():
            text = re.sub(r'\b{}\b'.format(re.escape(other)), key, text)

        return text

    def preprocessing(self, 
                      stopword_flag=True, 
                      manufact_stopword_flag=True, 
                      multi_words_flag=True, 
                      replacement_flag = True,
                      save_file=True):
        """
        preprocessing method
        - data information extraction
        - manufacturer encoding
        - manufacturer stopword removal
        - multiple words merging
        - stopword removal
        - other word replacement
        """
        
        # Extract target columns and dropna
        if 'DESCRIPTION OF FACTS CAUSING DISENGAGEMENT' in self.data.columns:
            self.data = self.data.dropna(subset=['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'])
        else:
            print("Please check dataset columns - Not exists 'DESCRIPTION OF FACTS CAUSING DISENGAGEMENT' column")
            return -1
        
        if 'DATE' in self.data.columns:
            self.data['DATE'] = pd.to_datetime(self.data['DATE'], format='mixed')
            self.data['year'] = self.data['DATE'].dt.strftime('%y')#year
            self.data['month'] = self.data['DATE'].dt.strftime('%m')
            self.data['day'] = self.data['DATE'].dt.strftime('%d')
            self.data['weekday'] = self.data['DATE'].dt.day_name()
            
        else:
            print("Please check dataset columns - Not exists 'DATE' column")
            return -1
        
        if 'Manufacturer' in self.data.columns:
            # Generate two-letter codes for each unique manufacturer        
            unique_manufacturers = self.data['Manufacturer'].unique()
            codes = [chr(first) + chr(second) for first in range(ord('A'), ord('Z')+1) for second in range(ord('A'), ord('Z')+1)]
            manufacturer_codes = dict(zip(unique_manufacturers, codes))

            # Map the codes to the 'Manufacturer' column
            self.data['Manufacturer'] = self.data['Manufacturer'].map(manufacturer_codes)
            
        else:
            print("Please check dataset columns - Not exists 'Manufacturer' column")
            return -1
        
        # Removel manufacturers' sentences in description columns (DESCRIPTION OF FACTS CAUSING DISENGAGEMENT)
        doc =  self.data['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT']

        # print("==================== origin descriptions ====================")
        # print(doc)

        if manufact_stopword_flag:
            manufact_stopword = self.manufact_stopword['stop_sentence'].tolist()
            doc = doc.apply(lambda x: self.__remove_manufact_stopwords(x, manufact_stopword))
            # print("==================== manufact_stopwords removal ====================")
            # print(doc)
            self.data['PREP1_MANUFACT_STOPWORD_REMOVAL'] = doc
        
        if multi_words_flag:
            multi_word_terms = self.multi_words['multi_words'].tolist()
            doc = doc.apply(lambda x: self.__replace_multi_words(x, multi_word_terms))
            # print("==================== multi_words replacement ====================")
            # print(doc)
            self.data['PREP2_MULTI_WORD_REPLACEMENT'] = doc
            
        if stopword_flag:
            stop_words = self.stopword['additional_stop_words'].tolist()
            doc = doc.apply(lambda x: self.__remove_general_stopwords(x, stop_words))
            # print("==================== general stopwords removal ====================")
            # print(doc)
            self.data['PREP3_GENERAL_STOPWORD_REMOVAL'] = doc
    
        if replacement_flag:
            replacements = {}
            for _, row in self.replacement_dict.iterrows():
                key_expression = row['key expression']
                for col in self.replacement_dict.columns:
                    if col != 'key expression' and pd.notna(row[col]):
                        replacements[row[col]] = key_expression

            doc = doc.apply(lambda x: self.__replace_general_form(x, replacements))
            # print("==================== general form replacement ====================")
            # print(doc)
            self.data['PREP4_TERM_FORM_REPLACEMENT'] = doc

        self.data['preprocessed'] = doc

        # save files
        directory, filename = os.path.split(self.data_path)        
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}-preprocessed{ext}"
        output_path = os.path.join(self.output_path, new_filename)
        self.data.to_csv(output_path)
        print(f"Preprocssed file is saved in {output_path}")

        new_filename = f"{name}-preprocessed_final{ext}"
        output_path = os.path.join(self.output_path, new_filename)
        self.data[['Manufacturer', 'year', 'preprocessed']].to_csv(output_path)
        print(f"Preprocssed file is saved (for analysis) in {output_path}")

    def data_sampling_for_expert_labeling(self):
        manu_list = list(set(self.data['Manufacturer']))        
        sampled_df = pd.DataFrame(self.data[self.data['Manufacturer'] == manu_list[0]].sample(20))
        sampled_df['Label'] = ''

        for i, manu in enumerate(manu_list):
            if i > 0:
                if len(self.data[self.data['Manufacturer'] == manu]) < 20:
                    sampled_df = pd.concat([sampled_df, self.data[self.data['Manufacturer'] == manu]], axis=0)                    
                else:
                    sampled_df = pd.concat([sampled_df, self.data[self.data['Manufacturer'] == manu].sample(20)], axis=0)

        # save file
        directory, filename = os.path.split(self.data_path)        
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}-for_domain_knowledge{ext}"
        output_path = os.path.join(self.output_path, new_filename)
        sampled_df[['Manufacturer', 'year', 'DESCRIPTION OF FACTS CAUSING DISENGAGEMENT', 'Label']].to_csv(output_path)
        print(f"Preprocssed file is saved in {output_path}")