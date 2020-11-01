import pickle
import pandas as pd
import os
import numpy as np
import multiprocessing as mp
from collections import Counter
# from RobEn.AggClust_new.ner import *
from RobEn.AggClust_new.agglomerative_clustering_new import *
# from agglomerative_clustering_new import *

K = 2
##############################################################
NUM_CORES = 1
cluster_graph = None
fields = None
all_right_fields = None
##############################################################
def encoding_training_set(fields, training_set):

    word_frequency_map = Counter()

    for index, row in training_set.iterrows():
        for field in fields:
            words_list = str(row[field]).split(' ')
            this_freq_map = Counter(words_list)
            word_frequency_map = word_frequency_map + this_freq_map
        # print(index)

    # with open(os.path.join(DATA_PATH, 'company_word_frequency_map.pickle'), 'wb') as f:
    #     pickle.dump(word_frequency_map, f)
    # with open(os.path.join(DATA_PATH, 'company_word_frequency_map_noemptystring.pickle'), 'rb') as f:
    #     word_frequency_map = pickle.load(f)

    remove_list = []

    for key in word_frequency_map:
        if word_frequency_map[key] < MIN_FREQUENCY:
            remove_list.append(key)

    for key in remove_list:
        word_frequency_map.pop(key)

    cluster_graph = cluster(word_frequency_map, GAMMA)
    # cluster_graph.print_graph()

    all_right_fields = [x for x in list(training_set)[2:] if x.startswith('right')]

    encoded_training_set = encode_parallel(training_set, cluster_graph, fields, all_right_fields)

    return encoded_training_set, cluster_graph


def encode_parallel(training_set, the_cluster_graph, the_fields, the_all_right_fields):
    global cluster_graph, fields, all_right_fields

    cluster_graph = the_cluster_graph
    fields = the_fields
    all_right_fields = the_all_right_fields

    df_chunks = np.array_split(training_set, NUM_CORES)

    with mp.Pool(NUM_CORES) as pool:
        processes_results = pool.map(encode_chunk, df_chunks)

        chunk_list = []

        for x in processes_results:
            chunk_list.append(x)

        processed_df = pd.concat(chunk_list, ignore_index=True)

    return processed_df


def encode_chunk(chunk):
    global fields, cluster_graph, all_right_fields
    # token_count = 0

    for index, row in chunk.iterrows():
        for field in fields:
            words_list = str(row[field]).split(' ')
            # token_count = token_count + len(words_list)
            encoded_words_list = [get_closet_encoding(item, cluster_graph, field, row, all_right_fields) for item in words_list]

            new_cell = ' '.join(map(str, encoded_words_list))
            chunk._set_value(index, field, new_cell)

        print(index)

    # print(token_count)
    return chunk


def encode(dataset, cluster_graph, fields, all_right_fields):
    print("K=", K)
    for index, row in dataset.iterrows():
        for field in fields:
            words_list = str(row[field]).split(' ')

            encoded_words_list = [get_closet_encoding(item, cluster_graph, field, row, all_right_fields) for item in words_list]

            new_cell = ' '.join(map(str, encoded_words_list))
            dataset._set_value(index, field, new_cell)

        # print(index)

    return dataset


def get_closet_encoding(word, cluster_graph, field, row, all_right_fields):
    encoded_word = cluster_graph.encode_word(word)

    if encoded_word is not None:
        return encoded_word

    if field.split('_')[0] == 'left':
        for x in all_right_fields:
            right_words_list = str(row[x]).split(' ')

            for right_word in right_words_list:
                if damerau_levenshtein_distance(right_word, word) <= K:
                    right_encoded_word = cluster_graph.encode_word(right_word)
                    if right_encoded_word is not None:
                        # print("a")
                        return right_encoded_word
                    else:
                        return right_word

        return word
    elif field.split('_')[0] == 'right':
        return word
##############################################################


if __name__ == "__main__":
    MIN_FREQUENCY = 1
    GAMMA = 1

    meta_data = {}

    DATA_PATH = '../../dataset/dirty/dirty_walmart_amazon_exp_data'
    TRAINING_SET = 'train.csv'
    # TRAINING_SET = 'train_tfidfv11.csv'

    # Beer
    # FIELDS = [('left_Beer_Name', 'right_Beer_Name'), ('left_Brew_Factory_Name', 'right_Brew_Factory_Name'), ('left_Style', 'right_Style')]
    # iTunes-Amazon
    # FIELDS = [('left_Song_Name', 'right_Song_Name'), ('left_Artist_Name', 'right_Artist_Name'), ('left_Album_Name', 'right_Album_Name'),
    #           ('left_Genre', 'right_Genre'), ('left_CopyRight', 'right_CopyRight'), ('left_Released', 'right_Released')]
    # Fodors-Zagat
    # FIELDS = [('left_name', 'right_name'), ('left_addr', 'right_addr'), ('left_city', 'right_city'), ('left_phone', 'right_phone'), ('left_type', 'right_type'), ('left_class', 'right_class')]
    # Walmart-Amazon
    FIELDS = [('left_title', 'right_title'), ('left_category', 'right_category'), ('left_brand', 'right_brand'), ('left_modelno', 'right_modelno')]
    # Amazon-Google
    # FIELDS = [('left_title', 'right_title'), ('left_manufacturer', 'right_manufacturer')]
    # DBLP-ACM
    # FIELDS = [('left_title', 'right_title'), ('left_authors', 'right_authors'), ('left_venue', 'right_venue'), ('left_year', 'right_year')]
    # DBLP-Scholar
    # FIELDS = [('left_title', 'right_title'), ('left_authors', 'right_authors'), ('left_venue', 'right_venue'), ('left_year', 'right_year')]
    # Abt-Buy
    # FIELDS = [('left_name', 'right_name'), ('left_description', 'right_description')]
    # Company
    # FIELDS = [('left_content', 'right_content')]

    meta_data['fields'] = FIELDS

    training_set = pd.read_csv(os.path.join(DATA_PATH, TRAINING_SET))

    for x in FIELDS:
        training_set, cluster_graph = encoding_training_set(x, training_set)

        meta_data[x] = cluster_graph
        print("Encoding training set field", x,"done.")

    training_set.to_csv(os.path.join(DATA_PATH, 'encoded_train_' + str(GAMMA) + '-' + str(MIN_FREQUENCY) + '.csv'), index=False)

    with open(os.path.join(DATA_PATH, 'encoded_cluster_' + str(GAMMA) + '-' + str(MIN_FREQUENCY) + '.pickle'), 'wb') as f:
        pickle.dump(meta_data, f)
