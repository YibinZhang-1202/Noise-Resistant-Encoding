import pandas as pd
import os
import numpy as np
import multiprocessing as mp
import nltk
from nltk import sent_tokenize, PorterStemmer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords


DATASET_PATH = 'dataset/textual/company_exp_data'
ORIGINAL_DATASET = 'train.csv'
GENERATING_SET = 'train_tfidfv12.csv'

MAX_LENGTH = 225
NUM_CORES = 16

all_content = ['left_content', 'right_content']


def process_chunk_function(chunk):
    for index, row in chunk.iterrows():
        print("Example ", index)

        for x in all_content:
            content = row[x]

            if content == '' or str(content) == 'nan' or len(content) == 1:
                continue

            sentences = sent_tokenize(content)

            vectorizer = CountVectorizer()
            spliter = vectorizer.build_tokenizer()

            freq_matrix = vectorizer.fit_transform(sentences)

            word = vectorizer.get_feature_names()

            transformer = TfidfTransformer()
            tfidf = transformer.fit_transform(freq_matrix)

            result = ''
            threshold = 0.0
            first_while = True
            while first_while or len(result.split(' ')) > MAX_LENGTH:
                first_while = False
                result = ''

                for c in range(len(sentences)):
                    sentence_split = spliter(sentences[c])

                    index_list = []
                    for a in sentence_split:
                        index_list.append(word.index(a))

                    tfidf_list = tfidf[c].toarray()[0][index_list]
                    for a, b in enumerate(sentence_split):
                        if tfidf_list[a] > threshold and b not in stopWords and b not in result.split():
                            result = result + b + ' '

                threshold += 0.1

            # print(len(result.split(' ')))
            chunk._set_value(index, x, result)

    return chunk


nltk.download('punkt')
nltk.download('stopwords')

dataset = pd.read_csv(os.path.join(DATASET_PATH, ORIGINAL_DATASET))
df_chunks = np.array_split(dataset, NUM_CORES)


stopWords = set(stopwords.words("english")).union(stop_words.ENGLISH_STOP_WORDS)

with mp.Pool(NUM_CORES) as pool:
    processed_df = pd.concat(pool.map(process_chunk_function, df_chunks), ignore_index=True)


processed_df.to_csv(os.path.join(DATASET_PATH, GENERATING_SET), index=False)

